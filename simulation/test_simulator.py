"""Tests for the simulator — the one call the other modules build on."""

from datetime import datetime

import pytest

from shared.schemas import AgentDecision, AgentRole, HouseholdProfile, OrderSide
from simulation.simulator import EPOCH, HouseholdSeries, MarketSimulator

IMPORT_TARIFF = 0.25
EXPORT_TARIFF = 0.07


def profile(
    household_id: str,
    role: AgentRole = AgentRole.PROSUMER,
    battery_capacity_kwh: float = 0.0,
) -> HouseholdProfile:
    """A household on the standard tariffs these tests use, batteries off by default."""
    return HouseholdProfile(
        household_id=household_id,
        role=role,
        has_solar=role is not AgentRole.PURE_CONSUMER,
        solar_capacity_kw=3.0 if role is not AgentRole.PURE_CONSUMER else 0.0,
        battery_capacity_kwh=battery_capacity_kwh,
        grid_export_tariff_eur_per_kwh=EXPORT_TARIFF,
        grid_import_tariff_eur_per_kwh=IMPORT_TARIFF,
    )


def order(
    household_id: str,
    tick: int,
    side: OrderSide,
    quantity_kwh: float,
    limit_price_eur_per_kwh: float,
) -> AgentDecision:
    return AgentDecision(
        household_id=household_id,
        tick=tick,
        side=side,
        quantity_kwh=quantity_kwh,
        limit_price_eur_per_kwh=limit_price_eur_per_kwh,
        strategy_name="test_fixture",
    )


def complementary_pair(ticks: int = 4, battery_capacity_kwh: float = 0.0):
    """A producer with 2.0 kWh spare a tick and a consumer 2.0 kWh short a tick."""
    return [
        HouseholdSeries(
            profile=profile("hh_sell", AgentRole.PURE_PRODUCER, battery_capacity_kwh),
            demand_kwh=[0.2] * ticks,
            solar_kwh=[2.2] * ticks,
        ),
        HouseholdSeries(
            profile=profile("hh_buy", AgentRole.PURE_CONSUMER),
            demand_kwh=[2.0] * ticks,
            solar_kwh=[0.0] * ticks,
        ),
    ]


def crossing_book(tick: int, quantity_kwh: float = 2.0):
    """A book that clears at the 0.15 midpoint of 0.10 and 0.20."""
    return [
        order("hh_sell", tick, OrderSide.SELL, quantity_kwh, 0.10),
        order("hh_buy", tick, OrderSide.BUY, quantity_kwh, 0.20),
    ]


def test_one_tick_end_to_end_by_hand():
    """Physics, clearing and settlement in one call, calculated by hand.

    The producer has 2.0 kWh spare and the consumer is 2.0 kWh short. They clear
    2.0 kWh at the 0.15 midpoint, so the buyer pays 0.30 instead of 2.0 * 0.25 = 0.50
    and the seller earns 0.30 instead of 2.0 * 0.07 = 0.14.
    """
    simulator = MarketSimulator(complementary_pair())

    result = simulator.step(crossing_book(tick=0))

    assert result.tick == 0
    assert result.timestamp == EPOCH
    assert len(result.market_state.trades) == 1
    assert result.market_state.clearing_price_eur_per_kwh == pytest.approx(0.15)

    assert result.household_states["hh_sell"].net_position_kwh == pytest.approx(2.0)
    assert result.household_states["hh_buy"].net_position_kwh == pytest.approx(-2.0)

    buyer = result.settlements["hh_buy"]
    assert buyer.p2p_cost_eur == pytest.approx(0.30)
    assert buyer.grid_only_cost_eur == pytest.approx(0.50)
    assert buyer.savings_eur == pytest.approx(0.20)

    seller = result.settlements["hh_sell"]
    assert seller.p2p_cost_eur == pytest.approx(-0.30)
    assert seller.savings_eur == pytest.approx(0.16)

    assert result.savings_eur == pytest.approx(0.36)


def test_the_battery_absorbs_surplus_before_the_market_sees_it():
    """Storage is physics the environment runs, so agents trade only what is left.

    The producer's 2.0 kWh surplus goes into an empty 5.0 kWh battery, leaving it
    nothing to sell. Its 2.0 kWh sell order is therefore energy it does not have, and
    settling at the meter charges it for the shortfall instead of paying it.
    """
    simulator = MarketSimulator(complementary_pair(battery_capacity_kwh=5.0))

    result = simulator.step(crossing_book(tick=0))

    seller_state = result.household_states["hh_sell"]
    assert seller_state.battery_level_kwh == pytest.approx(2.0)
    assert seller_state.net_position_kwh == pytest.approx(0.0)

    seller = result.settlements["hh_sell"]
    assert seller.grid_fallback_kwh == pytest.approx(2.0)
    # Earned 0.30 for energy it stored instead, then imported 2.0 kWh at 0.25.
    assert seller.p2p_cost_eur == pytest.approx(-0.30 + 0.50)
    assert seller.savings_eur < 0


def test_a_battery_carries_charge_between_ticks():
    """A tick depends on every tick before it: charge stored now is spent later."""
    producer = HouseholdSeries(
        profile=profile("hh_solo", AgentRole.PROSUMER, battery_capacity_kwh=5.0),
        demand_kwh=[0.0, 1.5],
        solar_kwh=[2.0, 0.0],
    )
    simulator = MarketSimulator([producer])

    charged = simulator.step([])
    discharged = simulator.step([])

    assert charged.household_states["hh_solo"].battery_level_kwh == pytest.approx(2.0)
    assert discharged.household_states["hh_solo"].battery_level_kwh == pytest.approx(0.5)
    # The battery covered the whole deficit, so nothing was needed from the grid.
    assert discharged.household_states["hh_solo"].net_position_kwh == pytest.approx(0.0)
    assert discharged.settlements["hh_solo"].p2p_cost_eur == pytest.approx(0.0)


def test_a_quiet_tick_still_settles_every_household_at_the_grid():
    """No orders is not no energy — households import and export as usual, saving nothing."""
    simulator = MarketSimulator(complementary_pair())

    result = simulator.step([])

    assert result.market_state.trades == []
    assert result.market_state.clearing_price_eur_per_kwh is None
    assert result.settlements["hh_buy"].p2p_cost_eur == pytest.approx(2.0 * IMPORT_TARIFF)
    assert result.settlements["hh_sell"].p2p_cost_eur == pytest.approx(-2.0 * EXPORT_TARIFF)
    assert result.savings_eur == pytest.approx(0.0)


def test_trading_more_than_exists_loses_both_sides_money():
    """The exploit the metered settlement closes, through the simulator this time."""
    simulator = MarketSimulator(complementary_pair())

    result = simulator.step(crossing_book(tick=0, quantity_kwh=50.0))

    assert result.market_state.trades[0].quantity_kwh == pytest.approx(50.0)
    assert result.savings_eur < 0
    assert all(s.savings_eur < 0 for s in result.settlements.values())


def test_replaying_a_run_reproduces_it_exactly():
    """Reset replays from tick 0 with the starting charge — the RL team debugs on this."""
    simulator = MarketSimulator(complementary_pair(battery_capacity_kwh=1.0))

    first = [simulator.step(crossing_book(tick)) for tick in range(4)]
    simulator.reset()
    second = [simulator.step(crossing_book(tick)) for tick in range(4)]

    assert simulator.tick == 4
    assert [r.market_state.model_dump() for r in first] == [
        r.market_state.model_dump() for r in second
    ]
    assert [r.settlements for r in first] == [r.settlements for r in second]


def test_two_simulators_on_the_same_inputs_agree():
    """Determinism across instances, not just across replays of one."""
    left = MarketSimulator(complementary_pair())
    right = MarketSimulator(complementary_pair())

    assert left.step(crossing_book(0)).settlements == right.step(crossing_book(0)).settlements


def test_orders_from_the_wrong_tick_are_rejected():
    """A desynchronised caller is a bug to surface, not to paper over."""
    simulator = MarketSimulator(complementary_pair())

    with pytest.raises(ValueError, match="tick"):
        simulator.step(crossing_book(tick=3))


def test_stepping_past_the_end_of_the_run_is_an_error():
    simulator = MarketSimulator(complementary_pair(ticks=2))

    simulator.step([])
    simulator.step([])

    with pytest.raises(IndexError, match="reset"):
        simulator.step([])


def test_the_run_is_as_long_as_the_shortest_series():
    """Every tick needs every household, so the shortest series ends the run."""
    short, long = complementary_pair(ticks=6)
    short = HouseholdSeries(
        profile=short.profile, demand_kwh=short.demand_kwh[:3], solar_kwh=short.solar_kwh[:3]
    )

    assert MarketSimulator([short, long]).num_ticks == 3


def test_the_same_household_cannot_take_part_twice():
    producer, _ = complementary_pair()

    with pytest.raises(ValueError, match="twice"):
        MarketSimulator([producer, producer])


def test_a_simulator_needs_a_household():
    with pytest.raises(ValueError, match="at least one household"):
        MarketSimulator([])


def test_ticks_are_half_hourly_from_the_epoch():
    simulator = MarketSimulator(complementary_pair())

    simulator.step([])
    second = simulator.step([])

    assert second.timestamp == datetime(2026, 1, 1, 0, 30)


def test_batteries_can_start_part_charged():
    """Per-household starting charge, for resuming a run or seeding a scenario."""
    simulator = MarketSimulator(
        complementary_pair(battery_capacity_kwh=5.0),
        initial_battery_level_kwh={"hh_sell": 4.0},
    )

    result = simulator.step([])

    # 1.0 kWh of headroom left, so 1.0 kWh of the 2.0 kWh surplus reaches the market.
    assert result.household_states["hh_sell"].battery_level_kwh == pytest.approx(5.0)
    assert result.household_states["hh_sell"].net_position_kwh == pytest.approx(1.0)
