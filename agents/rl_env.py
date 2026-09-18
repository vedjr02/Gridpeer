"""Multi-agent RL environment: one agent per household, on the real marketplace.

Build-order step 2 (see agents/CLAUDE.md): an environment that follows PettingZoo's
Parallel API, so it is legible to anyone who has used the multi-agent RL ecosystem,
wrapping the same ``MarketSimulator`` the rest of the codebase runs on. Nothing here
re-implements the market — every tick is cleared by ``clear_tick`` and settled at
the meter by ``settle_tick`` — so a policy is trained against exactly the mechanism
it will be judged on.

**What an agent sees, does and is scored on**

- *Observation*: its own forecast (net position, demand, solar, confidence), the
  time of day, and its own grid tariffs. Deliberately nothing else: every feature
  is computable from the ``decide(forecast, profile)`` call the pipeline makes, so
  a trained policy deploys into ``dashboard.run_pipeline`` unchanged. Battery level
  is *not* observed — no shared contract carries it yet (see the open
  ``schema-change`` proposal) — which is why training defaults to batteries off.
- *Action*: one number in [-1, 1] that scales the offered quantity from 0% to 200%
  of the forecast position — a **correction to the rule-based baseline**, whose
  order is exactly the zero action (residual RL). Above 100% because the naive
  forecast lags the sun and under-predicts morning surplus. Price is *not* an
  action: every order is priced by the baseline's rule. The side is fixed by the
  sign of the forecast, as for the baseline.
- *Reward*: savings versus the grid-only counterfactual, settled against the
  **meter**, not the order — selling energy a household never had is bought back at
  the import tariff. By default (``reward_mode="mixed"``) each agent gets half its
  own savings plus half an equal share of the community's: the community share
  gives every household a stable collective signal, and the own share makes each
  household bear the cost of its own over-offering. ``"community"`` and
  ``"individual"`` are kept for comparison.

**Why quantity only** — measured on held-out data, not assumed:

1. Any *fixed* quantity/eagerness pair is at least 22% worse than the baseline: its
   size-dependent eagerness routes trades through households with the largest,
   most reliable positions. Starting a policy *at* the baseline (residual) spends
   training on the real headroom — forecast error, worth up to +56% against a
   perfect-foresight ceiling — instead of on rediscovering the yardstick.
2. With a price action and **individual** reward, the shared policy learned to offer
   200% of its forecast at the greediest price allowed. The auction splits each
   trade's surplus between buyer and seller, so every agent gained by claiming more
   of it; once all did, asks stopped meeting bids and trading collapsed (-88%).
3. With a price action and **community** reward, total savings rose (+9% mean over
   five seeds) — by moving the pure consumer's share to the sellers. Its savings
   fell from EUR 6.87 to EUR 0.69 over 20 unseen runs. A marketplace a consumer
   loses from is one it leaves.

Pricing only moves savings between neighbours; only volume creates them. So the
policy controls volume, and price stays with the baseline's even-split rule.

**Why a mixed reward.** Quantity has the same trap in milder form. In a midday
glut, over-buying lets more of a buyer's real deficit get matched; the excess is
re-exported at the export tariff, a loss for the buyer exactly equal to the seller's
gain. Community-neutral, so a **community** reward learned to offer 200% everywhere:
+11% total, stable — paid for by the pure consumer, down EUR 1.66 over 20 unseen
runs. **Individual** reward makes each household bear its own excess and found
+12% with nobody worse off on some seeds, but was unstable (-0.2% to +12.4%). The
50/50 mix trained stably positive on every seed (+6.9% to +12.7%) with the worst
household moving by at most EUR 0.17 over 20 runs. agents/train_rl.py picks among
seeds on a validation set, fairness first.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from agents.baseline import MIN_TRADEABLE_KWH, RuleBasedTrader
from forecasting.baseline import NaiveForecaster
from shared.schemas import AgentDecision, ForecastOutput, HouseholdProfile, OrderSide
from simulation.environment import HouseholdEnvironment
from simulation.simulator import HouseholdSeries, MarketSimulator, TickResult, timestamp_for

OBSERVATION_SIZE = 8
ACTION_SIZE = 1
# With battery control the agent also sees its state of charge and capacity, and
# chooses a target charge level.
BATTERY_OBSERVATION_SIZE = OBSERVATION_SIZE + 2
BATTERY_ACTION_SIZE = ACTION_SIZE + 1
# Battery capacities are a few kWh; this keeps the capacity feature near [0, 1].
_CAPACITY_SCALE_KWH = 5.0

# kWh per half-hour tick rarely exceeds ~2 for a residential household; clipping at
# five keeps a freak reading from dominating the network's input scale.
_KWH_CLIP = 5.0
# Tariffs are EUR/kWh, roughly 0.05-0.35; dividing by this keeps them near [0, 1].
_TARIFF_SCALE_EUR_PER_KWH = 0.35
# Rewards in cents rather than euros: per-tick savings are fractions of a cent, and
# PPO's value function trains poorly on targets that small.
REWARD_SCALE = 100.0
# The baseline whose price rule every order uses. Stateless, so one shared instance.
_BASELINE = RuleBasedTrader()


def build_observation(forecast: ForecastOutput, profile: HouseholdProfile) -> np.ndarray:
    """One agent's observation, from exactly what ``decide()`` is given.

    Shared by training and deployment so the two cannot drift: the features a
    policy learns on are, by construction, the features it is served.
    """
    hour = forecast.timestamp.hour + forecast.timestamp.minute / 60.0
    angle = 2.0 * math.pi * hour / 24.0
    return np.array(
        [
            np.clip(forecast.predicted_net_position_kwh, -_KWH_CLIP, _KWH_CLIP),
            np.clip(forecast.predicted_demand_kwh, 0.0, _KWH_CLIP),
            np.clip(forecast.predicted_solar_generation_kwh, 0.0, _KWH_CLIP),
            forecast.confidence,
            math.sin(angle),
            math.cos(angle),
            profile.grid_export_tariff_eur_per_kwh / _TARIFF_SCALE_EUR_PER_KWH,
            profile.grid_import_tariff_eur_per_kwh / _TARIFF_SCALE_EUR_PER_KWH,
        ],
        dtype=np.float32,
    )


def _order(
    net_kwh: float,
    action_scale: float,
    forecast: ForecastOutput,
    profile: HouseholdProfile,
    strategy_name: str,
) -> AgentDecision | None:
    """An order for ``net_kwh`` (+ sell, - buy), scaled by the quantity action, baseline-priced."""
    scale = 1.0 + float(np.clip(action_scale, -1.0, 1.0))
    quantity_kwh = abs(net_kwh) * scale
    if quantity_kwh < MIN_TRADEABLE_KWH:
        return None
    side = OrderSide.SELL if net_kwh > 0 else OrderSide.BUY
    return AgentDecision(
        household_id=forecast.household_id,
        tick=forecast.tick,
        side=side,
        quantity_kwh=quantity_kwh,
        limit_price_eur_per_kwh=_BASELINE._limit_price_eur_per_kwh(
            profile, side, abs(net_kwh)
        ),
        strategy_name=strategy_name,
    )


def decision_from_action(
    action: Sequence[float] | np.ndarray,
    forecast: ForecastOutput,
    profile: HouseholdProfile,
    strategy_name: str,
) -> AgentDecision | None:
    """Turn a policy's raw action into an order, or None to sit the tick out.

    ``action[0]`` in [-1, 1] scales the quantity to 0-200% of the forecast position;
    ``[0]`` reproduces the baseline's order exactly. The price is the baseline's,
    computed from the *forecast* position, so offering more never buys priority in
    the book. Out-of-range actions are clipped, never rejected — PPO's Gaussian
    policy samples outside the box.
    """
    return _order(
        forecast.predicted_net_position_kwh, action[0], forecast, profile, strategy_name
    )


def build_battery_observation(
    forecast: ForecastOutput, profile: HouseholdProfile, battery_level_kwh: float
) -> np.ndarray:
    """The base observation plus state of charge (0-1) and scaled capacity."""
    capacity_kwh = profile.battery_capacity_kwh
    state_of_charge = battery_level_kwh / capacity_kwh if capacity_kwh > 0 else 0.0
    return np.concatenate(
        [
            build_observation(forecast, profile),
            np.array(
                [state_of_charge, capacity_kwh / _CAPACITY_SCALE_KWH], dtype=np.float32
            ),
        ]
    )


def battery_decision(
    action: Sequence[float] | np.ndarray,
    forecast: ForecastOutput,
    profile: HouseholdProfile,
    environment: HouseholdEnvironment,
    strategy_name: str,
) -> tuple[AgentDecision | None, float]:
    """An order and a battery offset from a battery-controlling action.

    Residual, like the quantity action: ``[0, 0]`` is the rule-based baseline trading
    on top of the automatic battery, fully aware of it. ``action[1]`` in [-1, 1]
    adjusts the automatic battery by up to one tick's worth of power: positive
    discharges extra to sell to a neighbour, negative holds charge back for later.
    The household plans its order on the metered position that choice implies —
    the same physics the simulator then runs — and ``action[0]`` scales it.
    """
    max_flow_kwh = environment.max_flow_kwh
    if not math.isfinite(max_flow_kwh):
        max_flow_kwh = profile.battery_capacity_kwh
    offset_kwh = float(np.clip(action[1], -1.0, 1.0)) * max_flow_kwh
    planned_net_kwh = environment.planned_net_kwh(
        forecast.predicted_net_position_kwh, battery_offset_kwh=offset_kwh
    )
    order = _order(planned_net_kwh, action[0], forecast, profile, strategy_name)
    return order, offset_kwh


def status_quo_cost_eur(profile: HouseholdProfile, raw_net_kwh: float) -> float:
    """What a tick costs with no battery and no marketplace: all of it at grid tariffs.

    The fixed yardstick every strategy's bill is compared against. Settlement's own
    grid-only figure uses the metered position *after* the battery, so it cannot see
    what the battery earned; this one can. With no battery the two are identical.
    """
    if raw_net_kwh >= 0:
        return -raw_net_kwh * profile.grid_export_tariff_eur_per_kwh
    return -raw_net_kwh * profile.grid_import_tariff_eur_per_kwh


def without_battery(households: Sequence[HouseholdSeries]) -> list[HouseholdSeries]:
    """The same households with storage removed, matching the pipeline's default.

    The simulator runs whatever battery a profile declares; zeroing capacity is how
    the "batteries off" setting of ``dashboard.run_pipeline`` is reproduced here.
    """
    return [
        HouseholdSeries(
            profile=h.profile.model_copy(update={"battery_capacity_kwh": 0.0}),
            demand_kwh=h.demand_kwh,
            solar_kwh=h.solar_kwh,
        )
        for h in households
    ]


class GridPeerParallelEnv(ParallelEnv):
    """PettingZoo Parallel environment over one community of households.

    Each episode is ``episode_ticks`` long, starting at a tick drawn from the
    series (or at tick 0 when ``random_start`` is off). Forecasts are made from
    history *within the episode only*, so every episode starts cold, exactly as a
    ``run_pipeline`` run does — the policy never trains on a warm-up it will not get
    in deployment.
    """

    metadata = {"name": "gridpeer_parallel_v0", "render_modes": []}

    def __init__(
        self,
        households: Sequence[HouseholdSeries],
        episode_ticks: int | None = None,
        random_start: bool = True,
        use_battery: bool = False,
        forecast_window: int = 4,
        strategy_name: str = "rl_policy",
        reward_mode: str = "mixed",
        own_weight: float = 0.5,
        battery_control: bool = False,
        max_battery_power_kw: float | None = None,
        shaping_gamma: float | None = None,
    ) -> None:
        """households: the community and its full series (all the same length).

        episode_ticks: ticks per episode, default the whole series. random_start:
        draw each episode's start tick from the reset seed. use_battery: run the
        declared batteries; off by default because the policy cannot observe them.
        """
        if not households:
            raise ValueError("an environment needs at least one household")
        self._households = list(households) if use_battery else without_battery(households)
        series_ticks = min(len(h.demand_kwh) for h in self._households)
        self.episode_ticks = series_ticks if episode_ticks is None else episode_ticks
        if not 1 <= self.episode_ticks <= series_ticks:
            raise ValueError(
                f"episode_ticks must be in 1..{series_ticks} (got {self.episode_ticks})"
            )
        self._series_ticks = series_ticks
        if battery_control and not use_battery:
            raise ValueError("battery_control needs use_battery=True: nothing to control")
        self.battery_control = battery_control
        self.max_battery_power_kw = max_battery_power_kw
        # Potential-based shaping (Ng, Harada & Russell 1999) for battery control: a
        # kWh stored is credited at the household's mid-tariff when stored and debited
        # when used. Storing midday solar costs export revenue *now* and pays off hours
        # later; without shaping PPO sees the cost clearly and the payoff faintly, and
        # unlearns charging (measured: -70% against the automatic battery). Shaping of
        # this form provably leaves the optimal policy unchanged. It touches the
        # training reward only — ``infos["savings_eur"]`` stays the real saving.
        self.shaping_gamma = shaping_gamma
        if reward_mode not in ("community", "individual", "mixed"):
            raise ValueError(
                f"reward_mode must be 'community', 'individual' or 'mixed' (got {reward_mode})"
            )
        if not 0.0 <= own_weight <= 1.0:
            raise ValueError(f"own_weight must be in [0, 1] (got {own_weight})")
        self.reward_mode = reward_mode
        self.own_weight = own_weight
        self.random_start = random_start
        self.use_battery = use_battery
        self.strategy_name = strategy_name
        self._forecaster = NaiveForecaster(window=forecast_window)

        self.render_mode = None
        self.possible_agents = [h.profile.household_id for h in self._households]
        self.agents: list[str] = []
        self._profiles = {h.profile.household_id: h.profile for h in self._households}
        obs_size = BATTERY_OBSERVATION_SIZE if battery_control else OBSERVATION_SIZE
        act_size = BATTERY_ACTION_SIZE if battery_control else ACTION_SIZE
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_size,), dtype=np.float32)
        self._rng = np.random.default_rng()
        self._simulator: MarketSimulator | None = None
        self._window: list[HouseholdSeries] = []
        self._forecasts: dict[str, ForecastOutput] = {}
        self.last_tick: TickResult | None = None

    def observation_space(self, agent: str) -> spaces.Space:
        """Every household observes the same eight features."""
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space:
        """Every household chooses one quantity correction in [-1, 1]."""
        return self._action_space

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Start a new episode and return each household's first observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        latest_start = self._series_ticks - self.episode_ticks
        start = int(self._rng.integers(0, latest_start + 1)) if self.random_start else 0
        end = start + self.episode_ticks

        self._window = [
            HouseholdSeries(
                profile=h.profile,
                demand_kwh=list(h.demand_kwh[start:end]),
                solar_kwh=list(h.solar_kwh[start:end]),
            )
            for h in self._households
        ]
        self._simulator = MarketSimulator(
            self._window, max_battery_power_kw=self.max_battery_power_kw
        )
        self.agents = list(self.possible_agents)
        self.last_tick = None
        self._forecast_tick(0)
        return self._observations(), {agent: {} for agent in self.agents}

    def _forecast_tick(self, tick: int) -> None:
        """Forecast every household for ``tick`` from its history before it."""
        timestamp = timestamp_for(tick)
        self._forecasts = {
            h.profile.household_id: self._forecaster.forecast(
                household_id=h.profile.household_id,
                tick=tick,
                timestamp=timestamp,
                demand_history_kwh=h.demand_kwh[:tick],
                solar_history_kwh=h.solar_kwh[:tick],
            )
            for h in self._window
        }

    def battery_level_kwh(self, agent: str) -> float:
        """``agent``'s current battery charge, kWh."""
        assert self._simulator is not None
        return self._simulator.environments[agent].battery_level_kwh

    def _observations(self) -> dict[str, np.ndarray]:
        if self.battery_control:
            return {
                agent: build_battery_observation(
                    self._forecasts[agent], self._profiles[agent], self.battery_level_kwh(agent)
                )
                for agent in self.agents
            }
        return {
            agent: build_observation(self._forecasts[agent], self._profiles[agent])
            for agent in self.agents
        }

    def current_forecast(self, agent: str) -> ForecastOutput:
        """The forecast behind ``agent``'s current observation (for tests and baselines)."""
        return self._forecasts[agent]

    def step(self, actions: dict[str, np.ndarray]) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Submit every household's order, clear and settle the tick, and score it."""
        if self._simulator is None or not self.agents:
            raise RuntimeError("call reset() before step()")

        levels_before = {
            agent: self._simulator.environments[agent].battery_level_kwh for agent in self.agents
        }
        orders: list[AgentDecision] = []
        offsets: dict[str, float] = {}
        for agent in self.agents:
            if agent not in actions:
                continue
            if self.battery_control:
                order, offsets[agent] = battery_decision(
                    actions[agent],
                    self._forecasts[agent],
                    self._profiles[agent],
                    self._simulator.environments[agent],
                    self.strategy_name,
                )
            else:
                order = decision_from_action(
                    actions[agent], self._forecasts[agent], self._profiles[agent],
                    self.strategy_name,
                )
            if order is not None:
                orders.append(order)
        result = self._simulator.step(orders, battery_offsets_kwh=offsets or None)
        self.last_tick = result

        # Savings against the status quo (no battery, no market), so that what a
        # battery earns counts. With batteries off this equals settlement's own
        # savings figure exactly — the pipeline-equivalence test pins that.
        own_savings_eur = {}
        for agent in self.agents:
            state = result.household_states[agent]
            raw_net_kwh = state.solar_generation_kwh - state.demand_kwh
            bill_eur = (
                result.settlements[agent].p2p_cost_eur
                if agent in result.settlements
                else status_quo_cost_eur(self._profiles[agent], state.net_position_kwh)
            )
            own_savings_eur[agent] = (
                status_quo_cost_eur(self._profiles[agent], raw_net_kwh) - bill_eur
            )
        community_eur = sum(own_savings_eur.values())
        if self.reward_mode == "community":
            rewards = dict.fromkeys(self.agents, community_eur * REWARD_SCALE)
        elif self.reward_mode == "individual":
            rewards = {agent: own * REWARD_SCALE for agent, own in own_savings_eur.items()}
        else:
            # Own savings plus an equal share of the community's: the share keeps the
            # collective signal, the own term makes each household bear its own
            # over-offering instead of passing it to a neighbour.
            share_eur = community_eur / len(self.agents)
            rewards = {
                agent: (self.own_weight * own + (1.0 - self.own_weight) * share_eur)
                * REWARD_SCALE
                for agent, own in own_savings_eur.items()
            }
        done = self._simulator.tick >= self._simulator.num_ticks
        if self.battery_control and self.shaping_gamma is not None:
            for agent in self.agents:
                profile = self._profiles[agent]
                value = (
                    profile.grid_export_tariff_eur_per_kwh
                    + profile.grid_import_tariff_eur_per_kwh
                ) / 2.0
                # Phi(terminal) = 0: charge left at the end of a run earns nothing.
                level_after = 0.0 if done else result.household_states[agent].battery_level_kwh
                shaping_eur = (self.shaping_gamma * level_after - levels_before[agent]) * value
                rewards[agent] += shaping_eur * REWARD_SCALE
        # Each household's own savings, whatever it was rewarded on: summing these over
        # agents always gives the community total, so episode accounting never
        # double-counts a shared reward.
        infos = {agent: {"savings_eur": own_savings_eur[agent]} for agent in self.agents}

        if done:
            observations = {
                agent: np.zeros(OBSERVATION_SIZE, dtype=np.float32) for agent in self.agents
            }
            terminations = dict.fromkeys(self.agents, True)
            truncations = dict.fromkeys(self.agents, False)
            self.agents = []
            return observations, rewards, terminations, truncations, infos

        self._forecast_tick(self._simulator.tick)
        observations = self._observations()
        terminations = dict.fromkeys(self.agents, False)
        truncations = dict.fromkeys(self.agents, False)
        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        """No rendering: the dashboard is the view onto a run."""
        return None

    def close(self) -> None:
        """Nothing to release."""
        return None
