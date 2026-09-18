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
from datetime import datetime

from shared.schemas import HouseholdProfile
from shared.schemas import HouseholdState as SharedHouseholdState


def battery_flow_kwh(
    net_kwh: float,
    battery_level_kwh: float,
    battery_capacity_kwh: float,
    max_flow_kwh: float = float("inf"),
    battery_offset_kwh: float = 0.0,
    battery_setpoint_kwh: float | None = None,
) -> float:
    """Energy a battery moves in one tick, kWh: + out of the battery, - into it.

    The one definition of the battery physics. The simulator calls it with the
    tick's actual position; agents call it with a forecast to plan an order. So an
    agent's plan and what the meter then records can never disagree on the rules.

    Automatic mode charges from surplus and covers deficit; ``battery_offset_kwh``
    then discharges more (positive) or holds back / charges more (negative). A
    setpoint instead drives the charge toward a target. Always within capacity and
    the per-tick flow limit.
    """
    if battery_setpoint_kwh is not None:
        flow_kwh = battery_level_kwh - min(max(battery_setpoint_kwh, 0.0), battery_capacity_kwh)
    else:
        if net_kwh > 0:
            flow_kwh = -min(net_kwh, battery_capacity_kwh - battery_level_kwh)
        else:
            flow_kwh = min(-net_kwh, battery_level_kwh)
        flow_kwh += battery_offset_kwh
    flow_kwh = max(-max_flow_kwh, min(max_flow_kwh, flow_kwh))
    # Capacity bounds: can neither empty below zero nor fill above capacity.
    return max(battery_level_kwh - battery_capacity_kwh, min(battery_level_kwh, flow_kwh))


def max_flow_kwh_per_tick(max_power_kw: float | None, tick_minutes: int = 30) -> float:
    """A power limit in kW as energy per tick in kWh; unlimited when None."""
    return float("inf") if max_power_kw is None else max_power_kw * tick_minutes / 60.0


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
        max_battery_power_kw: float | None = None,
        tick_minutes: int = 30,
    ) -> None:
        """profile: the household's identity and tariffs.

        demand_kwh / solar_kwh: per-tick observed values in kWh, chronological and
        the same length. initial_battery_level_kwh: starting charge, within the
        profile's battery capacity. max_battery_power_kw: charge/discharge power limit,
        None for unlimited (the week-1 default, unchanged). tick_minutes: tick length,
        which turns that power limit into energy per tick.
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
        if max_battery_power_kw is not None and max_battery_power_kw < 0:
            raise ValueError(f"{profile.household_id}: max_battery_power_kw must be >= 0")
        self.initial_battery_level_kwh = initial_battery_level_kwh
        self.max_battery_power_kw = max_battery_power_kw
        # Energy the battery can move in one tick, kWh; infinite when unlimited.
        self.max_flow_kwh = max_flow_kwh_per_tick(max_battery_power_kw, tick_minutes)
        self.battery_level_kwh = initial_battery_level_kwh
        self.tick = 0

    def __len__(self) -> int:
        """Number of ticks this household has data for."""
        return len(self.demand_kwh)

    def reset(self) -> None:
        """Return to tick 0 with the starting battery charge."""
        self.battery_level_kwh = self.initial_battery_level_kwh
        self.tick = 0

    def _flow_kwh(
        self,
        raw_net_kwh: float,
        battery_setpoint_kwh: float | None = None,
        battery_offset_kwh: float = 0.0,
    ) -> float:
        """This household's battery flow this tick; see :func:`battery_flow_kwh`."""
        return battery_flow_kwh(
            raw_net_kwh,
            self.battery_level_kwh,
            self.profile.battery_capacity_kwh,
            self.max_flow_kwh,
            battery_offset_kwh,
            battery_setpoint_kwh,
        )

    def contract_state(
        self, timestamp: datetime, metered_net_position_kwh: float | None = None
    ) -> SharedHouseholdState:
        """This household's battery state at the start of the next tick, as the shared
        ``HouseholdState`` contract that agents receive."""
        return SharedHouseholdState(
            household_id=self.profile.household_id,
            tick=self.tick,
            timestamp=timestamp,
            battery_level_kwh=self.battery_level_kwh,
            battery_capacity_kwh=self.profile.battery_capacity_kwh,
            battery_max_power_kw=self.max_battery_power_kw,
            metered_net_position_kwh=metered_net_position_kwh,
        )

    def planned_net_kwh(
        self,
        forecast_net_kwh: float,
        battery_setpoint_kwh: float | None = None,
        battery_offset_kwh: float = 0.0,
    ) -> float:
        """The metered position this tick would give if the forecast came true.

        What a household can know before it trades: the battery's current charge and
        what it will command. Exactly the physics of :meth:`step`, run on a forecast.
        """
        return forecast_net_kwh + self._flow_kwh(
            forecast_net_kwh, battery_setpoint_kwh, battery_offset_kwh
        )

    def step(
        self,
        battery_setpoint_kwh: float | None = None,
        battery_offset_kwh: float = 0.0,
    ) -> HouseholdState:
        """Advance one tick and return the household's state for it.

        With no setpoint and no offset (the default), the battery runs automatically:
        surplus charges it before anything is offered to the market, and a deficit
        is drawn from it before anything is bought. Only what the battery cannot
        absorb or supply reaches the market.

        ``battery_offset_kwh`` adjusts that automatic behaviour: positive discharges
        more — stored energy to sell to a neighbour — and negative holds charge back
        or draws more in. ``battery_setpoint_kwh`` instead drives the charge toward a
        target level. Either way, charging beyond the household's own surplus is
        drawn from the grid through the meter, and everything stays within capacity
        and the power limit.
        """
        if self.tick >= len(self):
            raise IndexError(
                f"{self.profile.household_id}: no data for tick {self.tick} "
                f"(series covers ticks 0-{len(self) - 1})"
            )
        if battery_setpoint_kwh is not None and battery_offset_kwh:
            raise ValueError("give a battery setpoint or an offset, not both")

        tick = self.tick
        demand = self.demand_kwh[tick]
        solar = self.solar_kwh[tick]
        raw_net_kwh = solar - demand
        flow_kwh = self._flow_kwh(raw_net_kwh, battery_setpoint_kwh, battery_offset_kwh)
        self.battery_level_kwh -= flow_kwh

        self.tick += 1
        return HouseholdState(
            household_id=self.profile.household_id,
            tick=tick,
            demand_kwh=demand,
            solar_generation_kwh=solar,
            battery_level_kwh=self.battery_level_kwh,
            net_position_kwh=raw_net_kwh + flow_kwh,
        )

    def run(self, num_ticks: int | None = None) -> list[HouseholdState]:
        """Step from the current tick to ``num_ticks`` (default: the whole series)."""
        last_tick = len(self) if num_ticks is None else min(num_ticks, len(self))
        return [self.step() for _ in range(max(0, last_tick - self.tick))]
