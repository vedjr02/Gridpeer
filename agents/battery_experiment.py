"""Stage 5: can an agent that controls the battery beat the automatic one?

    python -m agents.battery_experiment      # train, select on validation, report on test

The automatic battery is self-consumption: surplus charges it, the household's own
deficit drains it. It never sells stored energy to a neighbour, so a producer with
little demand of its own fills its battery and leaves it full, and nobody's evening
peak gets touched. An agent that sets its battery's target charge can hold midday
solar and sell it into the evening, when buyers are short and the grid is under the
most strain.

**Everything is measured against the status quo** — no battery, no market, every
kWh at grid tariffs — because that is what a household has today, and because
settlement's own grid-only figure is taken after the battery and cannot see what the
battery earned. Five strategies on identical households, 2.5 kW battery power limit:

1. baseline, no battery (what the dashboard runs today);
2. baseline + automatic battery, *unaware* of it (the pipeline's ``use_battery=True``);
3. baseline + automatic battery, *aware* of it — orders only what the battery will
   not absorb or supply. The strong heuristic the RL policy has to beat;
4. ppo_v1 from Stage 4, no battery, if a trained model exists;
5. the battery-controlling RL policy.

**How it learns.** The battery action is a correction to the automatic battery
(zero = strategy 3 exactly), trained with potential-based reward shaping so that
energy stored at noon is credited when stored rather than hours later when it pays
off. Without shaping, PPO unlearned charging and fell 70-78% below strategy 3; with
it, it beats it. Each household is rewarded on its own savings: once shaping fixed
the credit assignment, individual reward beat the mixed reward on community savings
and peak reduction alike.

**What "fair" means here — a trade-off, reported, not hidden.** Controlling batteries
changes *who sells*: a producer's stored energy now competes for the consumer's
evening demand. So the policy cannot leave every household better off than under the
automatic battery — measured, the two prosumers give up part of their battery's gain
while the producer and consumer gain more. The rule applied is participation: a seed
is eligible only if *every household beats the status quo*, then the largest community
saving wins. The change against the automatic battery is printed per household so the
team can decide whether that is the right rule.

**Scope.** This runs inside ``agents/`` and ``simulation/`` only. Bringing the battery
policy into ``dashboard.run_pipeline`` needs agents to see and set the battery through
a shared contract — the open schema-change proposal. Until then this is the evidence
for it, not a deployed strategy.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from agents.baseline import MIN_TRADEABLE_KWH, RuleBasedTrader
from agents.rl_env import (
    GridPeerParallelEnv,
    battery_decision,
    build_battery_observation,
    status_quo_cost_eur,
)
from agents.rl_vec_env import SharedPolicyVecEnv
from agents.train_rl import (
    EPISODE_DAYS,
    TEST_SEED_BASE,
    TRAIN_DAYS,
    TRAIN_SEED,
    VALIDATION_SEED_BASE,
    cohort,
)
from forecasting.baseline import NaiveForecaster
from shared.schemas import AgentDecision, ForecastOutput
from simulation.simulator import HouseholdSeries, MarketSimulator, timestamp_for

MAX_BATTERY_POWER_KW = 2.5  # a typical home battery; 1.25 kWh per half-hour tick
BATTERY_MODEL_PATH = Path("models/ppo_battery_v1.zip")
BATTERY_STRATEGY_NAME = "ppo_battery_v1"
TIMESTEPS = 1_000_000
GAMMA = 0.99  # a charge at noon pays off hours later: the future matters now
COHORTS = 20

# decide(simulator, forecasts) -> (orders, battery offsets or None for automatic)
Decider = Callable[
    [MarketSimulator, dict[str, ForecastOutput]],
    tuple[list[AgentDecision], dict[str, float] | None],
]


@dataclass
class Outcome:
    """One strategy over one run, against the status quo."""

    savings_eur: float = 0.0
    peak_import_kwh: float = 0.0
    status_quo_peak_kwh: float = 0.0
    household_savings_eur: dict[str, float] = field(default_factory=dict)

    @property
    def peak_reduction_pct(self) -> float:
        """Drop in the community's highest single-tick grid import, versus the status quo."""
        if self.status_quo_peak_kwh <= 0:
            return 0.0
        return (self.status_quo_peak_kwh - self.peak_import_kwh) / self.status_quo_peak_kwh * 100


def rollout(
    households: Sequence[HouseholdSeries],
    decide: Decider,
    network_charge_eur_per_kwh: float = 0.0,
) -> Outcome:
    """Run one strategy over the households and score it against the status quo.

    ``network_charge_eur_per_kwh`` is levied on each P2P kWh (buyer pays); the status
    quo trades nothing, so it pays none.
    """
    simulator = MarketSimulator(
        households,
        max_battery_power_kw=MAX_BATTERY_POWER_KW,
        network_charge_eur_per_kwh=network_charge_eur_per_kwh,
    )
    forecaster = NaiveForecaster(window=4)
    profiles = simulator.profiles
    outcome = Outcome(household_savings_eur=dict.fromkeys(profiles, 0.0))

    for tick in range(simulator.num_ticks):
        forecasts = {
            h.profile.household_id: forecaster.forecast(
                household_id=h.profile.household_id,
                tick=tick,
                timestamp=timestamp_for(tick),
                demand_history_kwh=h.demand_kwh[:tick],
                solar_history_kwh=h.solar_kwh[:tick],
            )
            for h in households
        }
        orders, offsets = decide(simulator, forecasts)
        result = simulator.step(orders, battery_offsets_kwh=offsets)

        net_sold_kwh: dict[str, float] = defaultdict(float)
        for trade in result.market_state.trades:
            net_sold_kwh[trade.seller_id] += trade.quantity_kwh
            net_sold_kwh[trade.buyer_id] -= trade.quantity_kwh

        tick_import_kwh = tick_status_quo_import_kwh = 0.0
        for household_id, state in result.household_states.items():
            raw_net_kwh = state.solar_generation_kwh - state.demand_kwh
            saved = (
                status_quo_cost_eur(profiles[household_id], raw_net_kwh)
                - result.settlements[household_id].p2p_cost_eur
            )
            outcome.household_savings_eur[household_id] += saved
            outcome.savings_eur += saved
            imbalance_kwh = state.net_position_kwh - net_sold_kwh[household_id]
            tick_import_kwh += max(0.0, -imbalance_kwh)
            tick_status_quo_import_kwh += max(0.0, -raw_net_kwh)
        outcome.peak_import_kwh = max(outcome.peak_import_kwh, tick_import_kwh)
        outcome.status_quo_peak_kwh = max(outcome.status_quo_peak_kwh, tick_status_quo_import_kwh)
    return outcome


def without_batteries(households: Sequence[HouseholdSeries]) -> list[HouseholdSeries]:
    """The same households with storage removed."""
    return [
        HouseholdSeries(
            profile=h.profile.model_copy(update={"battery_capacity_kwh": 0.0}),
            demand_kwh=h.demand_kwh,
            solar_kwh=h.solar_kwh,
        )
        for h in households
    ]


def baseline_decider(aware: bool) -> Decider:
    """The rule-based baseline under the automatic battery, aware of it or not.

    Aware means it is handed the battery state (schema 0.2.0) — exactly what the
    dashboard pipeline now gives it — and plans on what the battery will leave.
    """
    trader = RuleBasedTrader()

    def decide(simulator, forecasts):
        orders = []
        for household_id, forecast in forecasts.items():
            state = (
                simulator.environments[household_id].contract_state(forecast.timestamp)
                if aware
                else None
            )
            order = trader.decide(forecast, simulator.profiles[household_id], state)
            if order is not None:
                orders.append(order)
        return orders, None

    return decide


def policy_decider(model: PPO, battery: bool, strategy_name: str) -> Decider:
    """A trained policy, with or without control of the battery."""
    from agents.rl_env import build_observation, decision_from_action

    def decide(simulator, forecasts):
        orders, offsets = [], {}
        for household_id, forecast in forecasts.items():
            profile = simulator.profiles[household_id]
            if battery:
                state = simulator.environments[household_id].contract_state(forecast.timestamp)
                obs = build_battery_observation(forecast, profile, state.battery_level_kwh)
                action, _ = model.predict(obs, deterministic=True)
                order, _ = battery_decision(action, forecast, profile, state, strategy_name)
                # The contract carries the battery adjustment on the order, so a
                # household that places no order leaves its battery automatic —
                # exactly as the dashboard pipeline will run it.
                if order is not None:
                    offsets[household_id] = order.battery_offset_kwh
            else:
                action, _ = model.predict(build_observation(forecast, profile), deterministic=True)
                order = decision_from_action(action, forecast, profile, strategy_name)
            if order is not None and order.quantity_kwh >= MIN_TRADEABLE_KWH:
                orders.append(order)
        return orders, (offsets or None)

    return decide


def train_battery_policy(seed: int, timesteps: int = TIMESTEPS) -> PPO:
    """Train one battery-controlling shared policy on the training split."""
    env = GridPeerParallelEnv(
        cohort(4, TRAIN_DAYS, TRAIN_SEED),
        episode_ticks=EPISODE_DAYS * 48,
        random_start=True,
        use_battery=True,
        battery_control=True,
        max_battery_power_kw=MAX_BATTERY_POWER_KW,
        shaping_gamma=GAMMA,
        reward_mode="individual",
    )
    model = PPO(
        "MlpPolicy",
        SharedPolicyVecEnv(env, seed=seed),
        n_steps=480,
        batch_size=480,
        gamma=GAMMA,
        ent_coef=0.0,
        # Start near the residual origin (the aware baseline) and explore gently:
        # a unit-std Gaussian on the battery offset thrashes the battery at random.
        policy_kwargs={"log_std_init": -1.0},
        seed=seed,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=timesteps)
    return model


def score(decide: Decider, seed_base: int, battery: bool, cohorts: int = COHORTS) -> Outcome:
    """Sum one strategy's outcome over ``cohorts`` runs (peaks: mean per run)."""
    total = Outcome(household_savings_eur=defaultdict(float))
    peaks, reductions = [], []
    for k in range(cohorts):
        households = cohort(4, EPISODE_DAYS, seed_base + k)
        run = rollout(households if battery else without_batteries(households), decide)
        total.savings_eur += run.savings_eur
        for household_id, saved in run.household_savings_eur.items():
            total.household_savings_eur[household_id] += saved
        peaks.append(run.peak_import_kwh)
        reductions.append(run.peak_reduction_pct)
    total.peak_import_kwh = float(np.mean(peaks))
    total.status_quo_peak_kwh = float(np.mean(reductions))  # reused: mean % reduction
    return total


def main() -> None:
    """Train seeds, select on validation, report every strategy once on test."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    args = parser.parse_args()

    candidates = []
    for seed in range(args.seeds):
        started = time.perf_counter()
        model = train_battery_policy(seed, args.timesteps)
        mine = score(policy_decider(model, True, BATTERY_STRATEGY_NAME),
                     VALIDATION_SEED_BASE, battery=True)
        worst_vs_status_quo = min(mine.household_savings_eur.values())
        candidates.append((seed, model, mine, worst_vs_status_quo))
        print(
            f"seed {seed}: {time.perf_counter() - started:.0f}s | validation savings "
            f"EUR {mine.savings_eur:.2f}, peak {-mine.status_quo_peak_kwh:+.1f}%, "
            f"worst household vs status quo EUR {worst_vs_status_quo:+.2f}"
        )

    eligible = [c for c in candidates if c[3] > 0]
    seed, model, _, _ = (
        max(eligible, key=lambda c: c[2].savings_eur) if eligible
        else max(candidates, key=lambda c: c[3])
    )
    print(f"selected seed {seed}"
          + ("" if eligible else " (no seed left every household ahead; least-bad chosen)"))
    BATTERY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(BATTERY_MODEL_PATH)

    strategies: list[tuple[str, Decider, bool]] = [
        ("baseline, no battery", baseline_decider(aware=False), False),
        ("baseline + auto battery, unaware", baseline_decider(aware=False), True),
        ("baseline + auto battery, aware", baseline_decider(aware=True), True),
    ]
    stage4 = Path("models/ppo_v1.zip")
    if stage4.exists():
        strategies.append(
            ("ppo_v1, no battery",
             policy_decider(PPO.load(stage4, device="cpu"), False, "ppo_v1"), False)
        )
    strategies.append(
        (BATTERY_STRATEGY_NAME, policy_decider(model, True, BATTERY_STRATEGY_NAME), True)
    )

    print(f"\nTEST (unseen): {COHORTS} runs x 4 households x {EPISODE_DAYS} days, "
          f"{MAX_BATTERY_POWER_KW} kW batteries — savings vs status quo (no battery, no market)")
    results = {name: score(decide, TEST_SEED_BASE, battery) for name, decide, battery in strategies}
    for name, result in results.items():
        print(f"  {name:<34} EUR {result.savings_eur:7.2f}   peak import "
              f"{result.peak_import_kwh:5.2f} kWh/tick ({-result.status_quo_peak_kwh:+5.1f}%)")

    auto = results["baseline + auto battery, aware"]
    rl = results[BATTERY_STRATEGY_NAME]
    print(f"\n{BATTERY_STRATEGY_NAME} vs aware automatic battery: "
          f"{(rl.savings_eur - auto.savings_eur) / abs(auto.savings_eur) * 100:+.1f}% savings")
    print(f"  {'household':<10} {'auto (vs status quo)':>21} {'RL (vs status quo)':>19} "
          f"{'RL - auto':>10}")
    for household_id in sorted(auto.household_savings_eur):
        a_eur = auto.household_savings_eur[household_id]
        r_eur = rl.household_savings_eur[household_id]
        print(f"  {household_id:<10} {a_eur:>21.2f} {r_eur:>19.2f} {r_eur - a_eur:>+10.2f}")
    print(f"saved policy to {BATTERY_MODEL_PATH}")


if __name__ == "__main__":
    main()
