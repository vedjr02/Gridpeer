"""Tests for the household environment and its battery accounting."""

import pytest

from shared.schemas import AgentRole, HouseholdProfile
from simulation.environment import HouseholdEnvironment


def profile(
    household_id: str = "hh_001",
    battery_capacity_kwh: float = 5.0,
    solar_capacity_kw: float = 3.0,
) -> HouseholdProfile:
    """A prosumer profile with the tariffs these tests don't vary."""
    return HouseholdProfile(
        household_id=household_id,
        role=AgentRole.PROSUMER,
        has_solar=solar_capacity_kw > 0,
        solar_capacity_kw=solar_capacity_kw,
        battery_capacity_kwh=battery_capacity_kwh,
        grid_export_tariff_eur_per_kwh=0.07,
        grid_import_tariff_eur_per_kwh=0.25,
    )


def test_surplus_charges_the_battery_before_reaching_the_market():
    """2.0 kWh surplus into an empty 5.0 kWh battery: nothing is offered to trade."""
    env = HouseholdEnvironment(profile(), demand_kwh=[1.0], solar_kwh=[3.0])

    state = env.step()

    assert state.battery_level_kwh == 2.0
    assert state.net_position_kwh == 0.0
    assert state.surplus_kwh == 0.0


def test_surplus_beyond_battery_capacity_is_offered_to_the_market():
    """A full battery cannot absorb any more, so the rest is tradeable surplus."""
    env = HouseholdEnvironment(
        profile(battery_capacity_kwh=1.0),
        demand_kwh=[1.0],
        solar_kwh=[4.0],
    )

    state = env.step()

    assert state.battery_level_kwh == 1.0
    assert state.net_position_kwh == pytest.approx(2.0)
    assert state.surplus_kwh == pytest.approx(2.0)


def test_deficit_is_drawn_from_the_battery_first():
    """A charged battery covers the shortfall, so the household buys nothing."""
    env = HouseholdEnvironment(
        profile(),
        demand_kwh=[2.0],
        solar_kwh=[0.0],
        initial_battery_level_kwh=3.0,
    )

    state = env.step()

    assert state.battery_level_kwh == 1.0
    assert state.net_position_kwh == 0.0
    assert state.deficit_kwh == 0.0


def test_deficit_beyond_the_battery_must_be_bought():
    env = HouseholdEnvironment(
        profile(),
        demand_kwh=[2.0],
        solar_kwh=[0.0],
        initial_battery_level_kwh=0.5,
    )

    state = env.step()

    assert state.battery_level_kwh == 0.0
    assert state.net_position_kwh == pytest.approx(-1.5)
    assert state.deficit_kwh == pytest.approx(1.5)


def test_a_household_with_no_battery_trades_its_whole_net_position():
    """PURE_CONSUMER-style: no storage, so everything hits the market."""
    env = HouseholdEnvironment(
        profile(battery_capacity_kwh=0.0),
        demand_kwh=[1.5, 0.5],
        solar_kwh=[0.0, 2.0],
    )

    deficit_tick, surplus_tick = env.run()

    assert deficit_tick.net_position_kwh == pytest.approx(-1.5)
    assert surplus_tick.net_position_kwh == pytest.approx(1.5)
    assert surplus_tick.battery_level_kwh == 0.0


def test_battery_charge_carries_across_ticks():
    """Charge stored in a sunny tick covers a later dark one — the point of storage."""
    env = HouseholdEnvironment(
        profile(battery_capacity_kwh=5.0),
        demand_kwh=[0.5, 0.5, 2.0],
        solar_kwh=[3.0, 0.0, 0.0],
    )

    charge, draw, drain = env.run()

    assert charge.battery_level_kwh == pytest.approx(2.5)
    assert draw.battery_level_kwh == pytest.approx(2.0)
    assert drain.battery_level_kwh == 0.0
    # Every tick was covered by sun or storage: nothing needed the market.
    assert [s.net_position_kwh for s in (charge, draw, drain)] == [0.0, 0.0, 0.0]


def test_reset_replays_the_series_identically():
    """Reproducibility: the same series from the same charge gives the same states."""
    env = HouseholdEnvironment(
        profile(),
        demand_kwh=[0.5, 2.0, 1.0],
        solar_kwh=[3.0, 0.0, 0.2],
    )

    first = env.run()
    env.reset()
    second = env.run()

    assert first == second


def test_stepping_past_the_end_of_the_series_is_an_error():
    env = HouseholdEnvironment(profile(), demand_kwh=[1.0], solar_kwh=[0.0])
    env.step()

    with pytest.raises(IndexError, match="no data for tick"):
        env.step()


def test_mismatched_series_lengths_are_rejected():
    with pytest.raises(ValueError, match="same ticks"):
        HouseholdEnvironment(profile(), demand_kwh=[1.0, 1.0], solar_kwh=[0.0])


def test_negative_readings_are_rejected():
    with pytest.raises(ValueError, match="negative"):
        HouseholdEnvironment(profile(), demand_kwh=[-1.0], solar_kwh=[0.0])


def test_initial_battery_level_must_fit_the_profile():
    with pytest.raises(ValueError, match="outside the profile"):
        HouseholdEnvironment(
            profile(battery_capacity_kwh=2.0),
            demand_kwh=[1.0],
            solar_kwh=[0.0],
            initial_battery_level_kwh=5.0,
        )


# ---------------------------------------------------------------------------
# Battery setpoints and power limits
# ---------------------------------------------------------------------------


def test_no_setpoint_behaves_exactly_as_before():
    """Passing None is the automatic battery, bit for bit."""
    series = dict(demand_kwh=[0.5, 2.0, 1.0], solar_kwh=[3.0, 0.0, 0.2])
    automatic = HouseholdEnvironment(profile(), **series).run()
    explicit = HouseholdEnvironment(profile(), **series)
    assert [explicit.step(None) for _ in range(3)] == automatic


def test_holding_charge_sends_midday_surplus_to_the_market():
    """Setpoint = current level: the battery neither charges nor discharges.

    Hand-worked: 3.0 solar - 1.0 demand = +2.0 kWh raw. With the battery held at
    its current 1.0 kWh, all 2.0 kWh are offered to the market.
    """
    env = HouseholdEnvironment(
        profile(), demand_kwh=[1.0], solar_kwh=[3.0], initial_battery_level_kwh=1.0
    )
    state = env.step(battery_setpoint_kwh=1.0)
    assert state.battery_level_kwh == 1.0
    assert state.net_position_kwh == pytest.approx(2.0)


def test_discharging_into_the_evening_adds_energy_to_sell():
    """Hand-worked: 0.5 kWh demand, no sun, battery 4.0 -> 1.0 kWh.

    3.0 kWh flow out: 0.5 covers the home, 2.5 kWh is left to sell to neighbours.
    That is the evening-peak energy the automatic battery could never offer.
    """
    env = HouseholdEnvironment(
        profile(), demand_kwh=[0.5], solar_kwh=[0.0], initial_battery_level_kwh=4.0
    )
    state = env.step(battery_setpoint_kwh=1.0)
    assert state.battery_level_kwh == pytest.approx(1.0)
    assert state.net_position_kwh == pytest.approx(2.5)


def test_charging_beyond_own_surplus_draws_from_the_grid():
    """Setpoint above what solar provides: the difference is imported."""
    env = HouseholdEnvironment(profile(), demand_kwh=[1.0], solar_kwh=[1.5])
    state = env.step(battery_setpoint_kwh=2.0)
    assert state.battery_level_kwh == pytest.approx(2.0)
    assert state.net_position_kwh == pytest.approx(0.5 - 2.0)


def test_setpoints_are_clamped_to_capacity():
    env = HouseholdEnvironment(profile(battery_capacity_kwh=5.0), demand_kwh=[0.0, 0.0],
                               solar_kwh=[10.0, 0.0])
    assert env.step(battery_setpoint_kwh=99.0).battery_level_kwh == 5.0
    assert env.step(battery_setpoint_kwh=-3.0).battery_level_kwh == 0.0


def test_power_limit_caps_energy_moved_per_tick():
    """2.5 kW for 30 minutes is at most 1.25 kWh in or out per tick, both modes."""
    env = HouseholdEnvironment(
        profile(battery_capacity_kwh=5.0),
        demand_kwh=[0.0, 0.0, 3.0],
        solar_kwh=[4.0, 0.0, 0.0],
        max_battery_power_kw=2.5,
    )
    automatic = env.step()
    assert automatic.battery_level_kwh == pytest.approx(1.25)
    assert automatic.net_position_kwh == pytest.approx(4.0 - 1.25)

    commanded = env.step(battery_setpoint_kwh=5.0)
    assert commanded.battery_level_kwh == pytest.approx(2.5)

    drained = env.step(battery_setpoint_kwh=0.0)
    assert drained.battery_level_kwh == pytest.approx(1.25)
    assert drained.net_position_kwh == pytest.approx(-3.0 + 1.25)


def test_energy_is_conserved_under_any_setpoint():
    """Solar - demand = metered net + energy stored, every tick, whatever is commanded."""
    env = HouseholdEnvironment(
        profile(battery_capacity_kwh=5.0),
        demand_kwh=[0.4, 1.8, 0.9, 2.2],
        solar_kwh=[2.6, 0.3, 1.1, 0.0],
        initial_battery_level_kwh=1.0,
        max_battery_power_kw=3.0,
    )
    level = env.battery_level_kwh
    for demand, solar, setpoint in zip([0.4, 1.8, 0.9, 2.2], [2.6, 0.3, 1.1, 0.0],
                                       [4.0, 0.5, None, 2.0], strict=True):
        state = env.step(battery_setpoint_kwh=setpoint)
        stored = state.battery_level_kwh - level
        assert solar - demand == pytest.approx(state.net_position_kwh + stored)
        level = state.battery_level_kwh


def test_zero_offset_is_the_automatic_battery():
    series = dict(demand_kwh=[0.5, 2.0, 1.0], solar_kwh=[3.0, 0.0, 0.2])
    automatic = HouseholdEnvironment(profile(), max_battery_power_kw=2.5, **series).run()
    offset = HouseholdEnvironment(profile(), max_battery_power_kw=2.5, **series)
    assert [offset.step(battery_offset_kwh=0.0) for _ in range(3)] == automatic


def test_positive_offset_discharges_stored_energy_to_sell():
    """Hand-worked: evening, 0.2 kWh demand, battery full at 5.0 kWh, 2.5 kW limit.

    Automatic covers the 0.2 kWh. An offset of +1.0 discharges 1.0 kWh more, which
    the meter shows as +1.0 kWh to sell — the producer's evening energy for a
    neighbour, which the automatic battery would have kept.
    """
    env = HouseholdEnvironment(
        profile(), demand_kwh=[0.2], solar_kwh=[0.0], initial_battery_level_kwh=5.0,
        max_battery_power_kw=2.5,
    )
    state = env.step(battery_offset_kwh=1.0)
    assert state.battery_level_kwh == pytest.approx(3.8)
    assert state.net_position_kwh == pytest.approx(1.0)


def test_negative_offset_holds_charge_back():
    """A deficit the battery would cover is left to the market; the charge is kept."""
    env = HouseholdEnvironment(
        profile(), demand_kwh=[1.0], solar_kwh=[0.0], initial_battery_level_kwh=3.0,
        max_battery_power_kw=2.5,
    )
    state = env.step(battery_offset_kwh=-1.0)
    assert state.battery_level_kwh == pytest.approx(3.0)
    assert state.net_position_kwh == pytest.approx(-1.0)


def test_offsets_respect_power_and_capacity():
    env = HouseholdEnvironment(
        profile(battery_capacity_kwh=5.0), demand_kwh=[0.0, 0.0], solar_kwh=[0.0, 0.0],
        initial_battery_level_kwh=0.5, max_battery_power_kw=2.5,
    )
    assert env.step(battery_offset_kwh=9.0).battery_level_kwh == pytest.approx(0.0)
    assert env.step(battery_offset_kwh=-9.0).battery_level_kwh == pytest.approx(1.25)


def test_setpoint_and_offset_together_are_rejected():
    env = HouseholdEnvironment(profile(), demand_kwh=[1.0], solar_kwh=[0.0])
    with pytest.raises(ValueError, match="not both"):
        env.step(battery_setpoint_kwh=1.0, battery_offset_kwh=0.5)


@pytest.mark.parametrize("offset", [-2.0, -0.5, 0.0, 0.7, 3.0])
def test_planning_on_the_true_position_predicts_the_meter_exactly(offset):
    """planned_net_kwh and step run the same physics: plan on the truth, get the truth."""
    demand, solar = [0.3, 1.9, 0.6], [2.4, 0.1, 0.6]
    env = HouseholdEnvironment(
        profile(), demand_kwh=demand, solar_kwh=solar, initial_battery_level_kwh=2.0,
        max_battery_power_kw=2.5,
    )
    for d, s in zip(demand, solar, strict=True):
        planned = env.planned_net_kwh(s - d, battery_offset_kwh=offset)
        assert env.step(battery_offset_kwh=offset).net_position_kwh == pytest.approx(planned)
