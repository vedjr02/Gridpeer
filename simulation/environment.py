"""Household environment: what each household actually has to trade this tick.

Build-order step 2/3 (see simulation/CLAUDE.md). Turns a ``HouseholdProfile`` plus
its demand and solar time series into per-tick state — demand, generation, battery
level, and the net position left over once the battery has absorbed what it can.

That last number is the point of this module. A household with 2 kWh of surplus and
an empty battery does not offer 2 kWh to the market; it stores what it can and
offers the rest. Getting this wrong inflates traded volume and with it every
savings figure downstream.

Battery model, stated plainly: charge and discharge are capped only by remaining
capacity and remaining charge, with no round-trip efficiency loss, no charge-rate
limit and no degradation. That is a deliberate week-1 simplification — real losses
are 5-10% round trip — and it is the first thing to revisit when the numbers need
to be defensible rather than merely consistent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from shared.schemas import HouseholdProfile


@dataclass(frozen=True)
class HouseholdState:
    """One household's physical state for one tick. Internal to simulation/.

    Not a shared contract: nothing outside this module consumes it. What crosses
    the module boundary is the AgentDecision the agents module builds and the
    MarketState this module returns.

    Sign convention: ``net_position_kwh`` is positive when the household has
    surplus to sell and negative when it needs to buy — the same convention as
    ``ForecastOutput.predicted_net_position_kwh``, so a forecast and an outcome
    can be compared directly.
    """

    household_id: str
    tick: int
    demand_kwh: float
    solar_generation_kwh: float
    battery_level_kwh: float
    net_position_kwh: float

    @property
    def surplus_kwh(self) -> float:
        """Energy available to sell this tick (0.0 if the household is short)."""
        return max(0.0, self.net_position_kwh)

    @property
    def deficit_kwh(self) -> float:
        """Energy the household must source this tick (0.0 if it has surplus)."""
        return max(0.0, -self.net_position_kwh)


class HouseholdEnvironment:
    """Steps one household through its demand/solar series, tick by tick.

    Stateful by necessity — the battery carries charge between ticks, so a tick's
    net position depends on every tick before it. Call :meth:`step` in order from
    tick 0, or :meth:`reset` to replay from the start. Replaying the same series
    from the same starting charge always produces the same states; the agents
    module depends on that to reproduce a run it is debugging.
    """

    def __init__(
        self,
        profile: HouseholdProfile,
        demand_kwh: Sequence[float],
        solar_kwh: Sequence[float],
        initial_battery_level_kwh: float = 0.0,
    ) -> None:
        """profile: the household's identity and tariffs.

        demand_kwh / solar_kwh: per-tick observed values in kWh, chronological and
        the same length. initial_battery_level_kwh: starting charge, within the
        profile's battery capacity.
        """
        if len(demand_kwh) != len(solar_kwh):
            raise ValueError(
                f"{profile.household_id}: demand series has {len(demand_kwh)} ticks but "
                f"solar series has {len(solar_kwh)}; they must cover the same ticks"
            )
        if any(value < 0 for value in demand_kwh):
            raise ValueError(f"{profile.household_id}: demand_kwh contains a negative value")
        if any(value < 0 for value in solar_kwh):
            raise ValueError(f"{profile.household_id}: solar_kwh contains a negative value")
        if not 0 <= initial_battery_level_kwh <= profile.battery_capacity_kwh:
            raise ValueError(
                f"{profile.household_id}: initial battery level "
                f"{initial_battery_level_kwh} kWh is outside the profile's "
                f"0-{profile.battery_capacity_kwh} kWh capacity"
            )

        self.profile = profile
        self.demand_kwh = list(demand_kwh)
        self.solar_kwh = list(solar_kwh)
        self.initial_battery_level_kwh = initial_battery_level_kwh
        self.battery_level_kwh = initial_battery_level_kwh
        self.tick = 0

    def __len__(self) -> int:
        """Number of ticks this household has data for."""
        return len(self.demand_kwh)

    def reset(self) -> None:
        """Return to tick 0 with the starting battery charge."""
        self.battery_level_kwh = self.initial_battery_level_kwh
        self.tick = 0

    def step(self) -> HouseholdState:
        """Advance one tick and return the household's state for it.

        Surplus charges the battery before anything is offered to the market;
        a deficit is drawn from the battery before anything is bought. Only what
        the battery cannot absorb or supply reaches the market.
        """
        if self.tick >= len(self):
            raise IndexError(
                f"{self.profile.household_id}: no data for tick {self.tick} "
                f"(series covers ticks 0-{len(self) - 1})"
            )

        tick = self.tick
        demand = self.demand_kwh[tick]
        solar = self.solar_kwh[tick]
        raw_net_kwh = solar - demand

        if raw_net_kwh > 0:
            headroom_kwh = self.profile.battery_capacity_kwh - self.battery_level_kwh
            charged_kwh = min(raw_net_kwh, headroom_kwh)
            self.battery_level_kwh += charged_kwh
            net_position_kwh = raw_net_kwh - charged_kwh
        else:
            deficit_kwh = -raw_net_kwh
            discharged_kwh = min(deficit_kwh, self.battery_level_kwh)
            self.battery_level_kwh -= discharged_kwh
            net_position_kwh = -(deficit_kwh - discharged_kwh)

        self.tick += 1
        return HouseholdState(
            household_id=self.profile.household_id,
            tick=tick,
            demand_kwh=demand,
            solar_generation_kwh=solar,
            battery_level_kwh=self.battery_level_kwh,
            net_position_kwh=net_position_kwh,
        )

    def run(self, num_ticks: int | None = None) -> list[HouseholdState]:
        """Step from the current tick to ``num_ticks`` (default: the whole series)."""
        last_tick = len(self) if num_ticks is None else min(num_ticks, len(self))
        return [self.step() for _ in range(max(0, last_tick - self.tick))]
