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
