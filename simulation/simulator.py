"""One tick of the whole marketplace: physics, clearing and settlement in one call.

This is the module's public entry point. `agents/` wraps it to build an RL
environment, `dashboard/` calls it to advance the simulation loop, and neither has
to know how clearing, the battery or settlement work — they hand it a tick's
`AgentDecision` book and get back a `TickResult`.

Having one entry point is the point. Clearing, stepping each household's physics and
settling are three steps that must happen in the same order, with the same tick, on
the same data, every time; two callers doing that by hand is two chances to disagree
about the order and produce two different headline numbers.

**Order within a tick.** Physics runs first: each household's demand, solar and
battery for the tick are what they are, whatever anyone bid. Then the book clears.
Then the tick is settled against the metered net position that the physics produced,
so whatever a household traded but could not deliver (or bought but did not need) is
bought or sold at its own grid tariffs. Orders are promises, meters are facts.

**The battery is environment-owned physics** (team decision, see simulation/README.md).
Surplus charges the battery before anything reaches the market, and a deficit draws
from it before anything is bought; agents trade only what the battery could not
absorb or supply. The alternative — charge/discharge as an agent action — is a
larger action space and a harder learning problem, and it can be adopted later
without changing this signature, because it would add a field to `AgentDecision`
rather than change `step`. Until then, no RL policy can control storage.

Determinism: same households, same series, same orders, same results. The agents
module relies on that to reproduce a run it is debugging.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from shared.schemas import AgentDecision, HouseholdProfile, MarketState
from simulation.environment import HouseholdEnvironment, HouseholdState
from simulation.market import clear_tick
from simulation.settlement import HouseholdSettlement, settle_tick

TICK_MINUTES = 30
"""Half-hourly, matching the resolution of the CER smart-meter data."""

EPOCH = datetime(2026, 1, 1, 0, 0)
"""Tick 0's wall-clock time. The simulator owns the clock so every module agrees."""


def timestamp_for(tick: int, epoch: datetime = EPOCH) -> datetime:
    """Wall-clock time of ``tick``, counting from ``epoch`` in half-hour steps."""
    return epoch + timedelta(minutes=TICK_MINUTES * tick)


@dataclass(frozen=True)
class HouseholdSeries:
    """A household's identity plus the demand and solar it actually sees, in kWh.

    The series are ground truth — what the meter records — not a forecast. Building
    them from real CER/PVGIS data or from a synthetic generator belongs to `data/`;
    this module only consumes them.
    """

    profile: HouseholdProfile
    demand_kwh: Sequence[float]
    solar_kwh: Sequence[float]


@dataclass(frozen=True)
class TickResult:
    """Everything one tick produced, for whoever asked for it.

    `market_state` is the shared contract other modules persist and display;
    `settlements` is what the tick cost each household versus the grid-only
    counterfactual; `household_states` is the physics behind those numbers, which is
    what an RL observation is built from.
    """

    tick: int
    timestamp: datetime
    market_state: MarketState
    settlements: dict[str, HouseholdSettlement]
    household_states: dict[str, HouseholdState]

    @property
    def savings_eur(self) -> float:
        """What the marketplace saved the community this tick, in EUR."""
        return sum(settlement.savings_eur for settlement in self.settlements.values())


class MarketSimulator:
    """Steps a community of households through a run, one tick at a time.

    Stateful: batteries carry charge between ticks, so a tick depends on every tick
    before it. Call :meth:`step` in order from tick 0, or :meth:`reset` to replay
    from the start with the same starting charge.
    """

    def __init__(
        self,
        households: Sequence[HouseholdSeries],
        initial_battery_level_kwh: Mapping[str, float] | float = 0.0,
        epoch: datetime = EPOCH,
        max_battery_power_kw: float | None = None,
    ) -> None:
        """households: the community, each with its own profile and series.

        initial_battery_level_kwh: starting charge, either one value for everyone or
        per household_id. epoch: wall-clock time of tick 0. max_battery_power_kw:
        battery power limit for every household, None for unlimited (the default).
        """
        if not households:
            raise ValueError("a simulator needs at least one household")

        self.epoch = epoch
        self.profiles: dict[str, HouseholdProfile] = {}
        self.environments: dict[str, HouseholdEnvironment] = {}

        for household in households:
            household_id = household.profile.household_id
            if household_id in self.profiles:
                raise ValueError(
                    f"household {household_id} appears twice; each household is "
                    f"one participant with one meter"
                )
            level_kwh = (
                initial_battery_level_kwh
                if isinstance(initial_battery_level_kwh, (int, float))
                else initial_battery_level_kwh.get(household_id, 0.0)
            )
            self.profiles[household_id] = household.profile
            self.environments[household_id] = HouseholdEnvironment(
                profile=household.profile,
                demand_kwh=household.demand_kwh,
                solar_kwh=household.solar_kwh,
                initial_battery_level_kwh=float(level_kwh),
                max_battery_power_kw=max_battery_power_kw,
                tick_minutes=TICK_MINUTES,
            )

        # A tick needs every household, so the run is as long as the shortest series.
        self.num_ticks = min(len(env) for env in self.environments.values())
        self.tick = 0

    @property
    def household_ids(self) -> list[str]:
        """Participating households, in the order they were given."""
        return list(self.profiles)

    def reset(self) -> None:
        """Return every household to tick 0 with its starting battery charge."""
        for environment in self.environments.values():
            environment.reset()
        self.tick = 0

    def step(
        self,
        orders: Sequence[AgentDecision],
        battery_setpoints_kwh: Mapping[str, float] | None = None,
        battery_offsets_kwh: Mapping[str, float] | None = None,
    ) -> TickResult:
        """Advance one tick: run the physics, clear ``orders``, settle at the meter.

        ``orders`` is this tick's book, at most one per household — whatever the
        agents decided, including none at all. Their ``tick`` must be the tick being
        stepped; a mismatch is a desynchronised caller, not something to paper over.
        ``battery_setpoints_kwh`` drives named households' batteries toward a target
        charge this tick; ``battery_offsets_kwh`` adjusts their automatic behaviour
        instead (+ discharge more, - hold back). Everyone else's battery runs
        automatically, as before.
        """
        if self.tick >= self.num_ticks:
            raise IndexError(
                f"the run is over at tick {self.num_ticks}; call reset() to replay it"
            )

        tick = self.tick
        timestamp = timestamp_for(tick, self.epoch)

        # Physics first: what each household generated, used and stored this tick is
        # true regardless of what anyone offered the market.
        setpoints = battery_setpoints_kwh or {}
        offsets = battery_offsets_kwh or {}
        household_states = {
            household_id: environment.step(
                setpoints.get(household_id), offsets.get(household_id, 0.0)
            )
            for household_id, environment in self.environments.items()
        }

        market_state = clear_tick(tick=tick, timestamp=timestamp, orders=orders)
        settlements = settle_tick(
            market_state,
            self.profiles,
            actual_net_position_kwh={
                household_id: state.net_position_kwh
                for household_id, state in household_states.items()
            },
        )

        self.tick = tick + 1
        return TickResult(
            tick=tick,
            timestamp=timestamp,
            market_state=market_state,
            settlements=settlements,
            household_states=household_states,
        )
