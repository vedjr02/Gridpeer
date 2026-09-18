"""Tests for the rule-based baseline trader."""

from datetime import datetime

import pytest

from agents.baseline import STRATEGY_NAME, RuleBasedTrader
from shared.schemas import AgentDecision, AgentRole, ForecastOutput, HouseholdProfile, OrderSide

TICK = 3
TIMESTAMP = datetime(2026, 1, 1, 9, 30)
EXPORT_TARIFF = 0.07
IMPORT_TARIFF = 0.25


def profile(household_id: str = "hh_001") -> HouseholdProfile:
    return HouseholdProfile(
        household_id=household_id,
        role=AgentRole.PROSUMER,
        has_solar=True,
        solar_capacity_kw=3.0,
        battery_capacity_kwh=5.0,
        grid_export_tariff_eur_per_kwh=EXPORT_TARIFF,
        grid_import_tariff_eur_per_kwh=IMPORT_TARIFF,
    )


def forecast(
    net_position_kwh: float,
    household_id: str = "hh_001",
    confidence: float = 0.8,
) -> ForecastOutput:
    """A forecast with the requested net position, demand/solar made consistent with it."""
    demand = 1.0
    solar = demand + net_position_kwh
    return ForecastOutput(
        household_id=household_id,
        tick=TICK,
        timestamp=TIMESTAMP,
        predicted_demand_kwh=demand,
        predicted_solar_generation_kwh=max(0.0, solar),
        predicted_net_position_kwh=net_position_kwh,
        confidence=confidence,
    )


def test_surplus_becomes_a_sell_order():
    decision = RuleBasedTrader().decide(forecast(1.5), profile())

    assert decision is not None
    assert decision.side is OrderSide.SELL
    assert decision.quantity_kwh == pytest.approx(1.5)
    assert decision.strategy_name == STRATEGY_NAME
    assert decision.tick == TICK


def test_deficit_becomes_a_buy_order():
    decision = RuleBasedTrader().decide(forecast(-0.8), profile())

    assert decision is not None
    assert decision.side is OrderSide.BUY
    assert decision.quantity_kwh == pytest.approx(0.8)


def test_a_balanced_household_sits_the_tick_out():
    """Zero net position means no order at all — not a zero-quantity one.

    AgentDecision requires a positive quantity, so the only way to express
    "nothing to trade" is to not trade.
    """
    assert RuleBasedTrader().decide(forecast(0.0), profile()) is None
    assert RuleBasedTrader().decide(forecast(0.0001), profile()) is None


def test_prices_always_land_inside_the_tariff_gap():
    """Outside the gap, trading is strictly worse than using the grid.

    A sell below the export tariff earns less than exporting; a buy above the
    import tariff costs more than importing. Either makes the baseline worse than
    no marketplace at all, which would invalidate every comparison against it.
    """
    trader = RuleBasedTrader()

    for net_position in (0.1, 0.5, 2.0, 10.0, -0.1, -0.5, -2.0, -10.0):
        decision = trader.decide(forecast(net_position), profile())
        assert decision is not None
        assert EXPORT_TARIFF <= decision.limit_price_eur_per_kwh <= IMPORT_TARIFF


def test_default_eagerness_prices_a_small_order_at_the_midpoint():
    """Hand-calculated: base eagerness 0.5 and negligible urgency give the midpoint.

    Gap is 0.25 - 0.07 = 0.18. A tiny order carries urgency ~0, so eagerness is
    0.5 and the price is 0.07 + 0.5 * 0.18 = 0.16 from either side.
    """
    trader = RuleBasedTrader(urgency_gain=0.0)

    sell = trader.decide(forecast(0.01), profile())
    buy = trader.decide(forecast(-0.01), profile())

    assert sell is not None and buy is not None
    assert sell.limit_price_eur_per_kwh == pytest.approx(0.16)
    assert buy.limit_price_eur_per_kwh == pytest.approx(0.16)


def test_a_bigger_position_concedes_more_margin():
    """Urgency: the more energy at stake, the more price the household gives up."""
    trader = RuleBasedTrader()

    small_sell = trader.decide(forecast(0.2), profile())
    large_sell = trader.decide(forecast(5.0), profile())
    assert large_sell.limit_price_eur_per_kwh < small_sell.limit_price_eur_per_kwh

    small_buy = trader.decide(forecast(-0.2), profile())
    large_buy = trader.decide(forecast(-5.0), profile())
    assert large_buy.limit_price_eur_per_kwh > small_buy.limit_price_eur_per_kwh


def test_two_baseline_households_produce_a_crossing_book():
    """The baseline must actually trade with itself, or it flatters everything else.

    A seller's ask has to land at or below a buyer's bid; otherwise the yardstick
    never trades and any RL policy looks good by comparison.
    """
    trader = RuleBasedTrader()
    ask = trader.decide(forecast(1.0, household_id="hh_sell"), profile("hh_sell"))
    bid = trader.decide(forecast(-1.0, household_id="hh_buy"), profile("hh_buy"))

    assert ask.limit_price_eur_per_kwh <= bid.limit_price_eur_per_kwh


def test_decisions_are_reproducible():
    trader = RuleBasedTrader()
    same_forecast = forecast(1.25)

    first = trader.decide(same_forecast, profile())
    second = trader.decide(same_forecast, profile())

    assert first.model_dump() == second.model_dump()


def test_output_satisfies_the_shared_contract():
    decision = RuleBasedTrader().decide(forecast(1.5), profile())

    assert AgentDecision.model_validate(decision.model_dump()) == decision


def test_mismatched_forecast_and_profile_are_rejected():
    with pytest.raises(ValueError, match="hh_other"):
        RuleBasedTrader().decide(forecast(1.0, household_id="hh_other"), profile("hh_001"))


def test_base_eagerness_below_half_is_rejected():
    """Below 0.5 asks sit above bids and the baseline silently stops trading."""
    with pytest.raises(ValueError, match="base_eagerness"):
        RuleBasedTrader(base_eagerness=0.4)
