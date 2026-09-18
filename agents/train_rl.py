"""Train the shared PPO policy and measure it against the baseline on unseen data.

    python -m agents.train_rl                 # 5 seeds, pick on validation, report on test
    python -m agents.train_rl --seeds 1       # one seed, quicker

**Honest evaluation is the point of this script.** Three disjoint data splits, all
synthetic series from ``data.synthetic`` with the data module's default households:

- *train* (seed 1000, 60 days): what the policy learns from, in 2-day episodes;
- *validation* (seeds 7000+): used only to choose between training seeds;
- *test* (seeds 5000+): touched once, at the end, for the reported number.

Both strategies are scored by ``agents.evaluation.run_strategy`` — the harness
pinned by tests to report exactly what the dashboard reports — on identical
households, 2 days per run from a cold start, as the dashboard runs.

**Choosing among seeds, fairness first.** PPO's outcome varies with its seed. A
seed is eligible only if no household ends up worse off than under the baseline on
validation; among eligible seeds the largest community uplift wins. If none is
eligible, the seed with the smallest worst-household loss wins, and the report
says so. The community total is not allowed to buy itself out of one neighbour's
pocket.

**Why the discount factor is zero.** With batteries off, a household's order this
tick cannot change any later tick: nothing is stored, forecasts are made from meter
history that trades do not alter, and the market clears afresh. Each tick is a
separate decision scored on its own settlement — a contextual bandit — so
``gamma=0`` is the correct setting, not a shortcut. Once agents control a battery
(see the schema-change proposal) actions carry forward and gamma must rise with it.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO

from agents.baseline import STRATEGY_NAME as BASELINE_NAME
from agents.baseline import RuleBasedTrader
from agents.evaluation import HouseholdSeries as HarnessSeries
from agents.evaluation import TradingStrategy, run_strategy
from agents.rl_env import GridPeerParallelEnv
from agents.rl_policy import DEFAULT_MODEL_PATH, DEFAULT_STRATEGY_NAME, PPOTrader
from agents.rl_vec_env import SharedPolicyVecEnv
from data.synthetic import default_profiles, generate_household_series, ticks_per_day
from simulation.simulator import HouseholdSeries

TRAIN_SEED = 1_000
VALIDATION_SEED_BASE = 7_000
TEST_SEED_BASE = 5_000
TRAIN_DAYS = 60
EPISODE_DAYS = 2  # matches the dashboard's default run: two days from a cold start
VALIDATION_COHORTS = 20
TEST_COHORTS = 20

# The recipe that trained stably positive on every seed (see agents/rl_env.py).
TIMESTEPS = 300_000
N_STEPS = 480  # per household per update: five episodes, 1,920 samples at 4 households
BATCH_SIZE = 480
ENT_COEF = 0.01


def cohort(num_households: int, days: int, seed: int) -> list[HouseholdSeries]:
    """A synthetic community: the data module's default profiles and generated series."""
    households = []
    for profile in default_profiles(num_households):
        demand, solar = generate_household_series(profile, days=days, seed=seed)
        households.append(HouseholdSeries(profile=profile, demand_kwh=demand, solar_kwh=solar))
    return households


def train(
    num_households: int = 4,
    timesteps: int = TIMESTEPS,
    seed: int = 0,
    gamma: float = 0.0,
) -> PPO:
    """Train one shared policy on ``TRAIN_DAYS`` of synthetic data, in 2-day episodes."""
    env = GridPeerParallelEnv(
        cohort(num_households, TRAIN_DAYS, TRAIN_SEED),
        episode_ticks=EPISODE_DAYS * ticks_per_day(),
        random_start=True,
    )
    model = PPO(
        "MlpPolicy",
        SharedPolicyVecEnv(env, seed=seed),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=gamma,
        ent_coef=ENT_COEF,
        seed=seed,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=timesteps)
    return model


@dataclass(frozen=True)
class Comparison:
    """Baseline versus policy over a set of unseen cohorts."""

    baseline_eur: list[float]
    policy_eur: list[float]
    household_change_eur: dict[str, float]

    @property
    def wins(self) -> int:
        """Cohorts where the policy saved the community more than the baseline."""
        return sum(p > b for b, p in zip(self.baseline_eur, self.policy_eur, strict=True))

    @property
    def uplift_pct(self) -> float:
        """Community savings uplift over the baseline, summed across cohorts."""
        base = sum(self.baseline_eur)
        return (sum(self.policy_eur) - base) / abs(base) * 100.0 if base else 0.0

    @property
    def worst_household_change_eur(self) -> float:
        """The largest loss (or smallest gain) any one household saw versus the baseline."""
        return min(self.household_change_eur.values())

    @property
    def fair(self) -> bool:
        """True when no household ends up worse off than under the baseline."""
        return self.worst_household_change_eur >= -1e-9


def evaluate(
    policy: TradingStrategy,
    num_households: int = 4,
    cohorts: int = TEST_COHORTS,
    seed_base: int = TEST_SEED_BASE,
    days: int = EPISODE_DAYS,
) -> Comparison:
    """Score baseline and policy on ``cohorts`` communities, per run and per household."""
    baseline_eur, policy_eur = [], []
    change: dict[str, float] = defaultdict(float)
    for k in range(cohorts):
        households = [
            HarnessSeries(h.profile, h.demand_kwh, h.solar_kwh)
            for h in cohort(num_households, days, seed_base + k)
        ]
        base = run_strategy(households, RuleBasedTrader(), f"base_{k}", BASELINE_NAME)
        mine = run_strategy(households, policy, f"rl_{k}", DEFAULT_STRATEGY_NAME)
        baseline_eur.append(base.summary.total_savings_eur)
        policy_eur.append(mine.summary.total_savings_eur)
        for outcome in base.outcomes:
            change[outcome.household_id] -= outcome.savings_eur
        for outcome in mine.outcomes:
            change[outcome.household_id] += outcome.savings_eur
    return Comparison(baseline_eur, policy_eur, dict(change))


def select(candidates: Sequence[tuple[int, PPO, Comparison]]) -> tuple[int, PPO, Comparison]:
    """Fairness first, then uplift — on validation results only."""
    fair = [c for c in candidates if c[2].fair]
    if fair:
        return max(fair, key=lambda c: c[2].uplift_pct)
    return max(candidates, key=lambda c: c[2].worst_household_change_eur)


def main() -> None:
    """Train each seed, select on validation, report once on test, save the policy."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=5, help="training seeds to try")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--households", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    candidates = []
    for seed in range(args.seeds):
        started = time.perf_counter()
        model = train(num_households=args.households, timesteps=args.timesteps, seed=seed)
        validation = evaluate(
            PPOTrader(model=model),
            num_households=args.households,
            cohorts=VALIDATION_COHORTS,
            seed_base=VALIDATION_SEED_BASE,
        )
        candidates.append((seed, model, validation))
        print(
            f"seed {seed}: trained in {time.perf_counter() - started:.0f}s | validation "
            f"uplift {validation.uplift_pct:+.1f}%, worst household "
            f"EUR {validation.worst_household_change_eur:+.3f}"
            f"{'' if validation.fair else '  (not fair)'}"
        )

    seed, model, validation = select(candidates)
    print(
        f"selected seed {seed}"
        + ("" if validation.fair else " — no seed was fair on validation; least-unfair chosen")
    )

    test = evaluate(PPOTrader(model=model), num_households=args.households)
    print(
        f"\nTEST (unseen): {TEST_COHORTS} runs x {args.households} households x "
        f"{EPISODE_DAYS} days"
    )
    print(f"  baseline  EUR {sum(test.baseline_eur):.3f} total")
    print(f"  ppo_v1    EUR {sum(test.policy_eur):.3f} total")
    print(f"  uplift    {test.uplift_pct:+.1f}%   (better in {test.wins}/{TEST_COHORTS} runs)")
    print("  per household vs baseline:")
    for household_id, change in sorted(test.household_change_eur.items()):
        print(f"    {household_id}: EUR {change:+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved policy to {args.out}")


if __name__ == "__main__":
    main()
