"""
Cross-module contract tests.

These don't test any module's logic — they test that the shared contracts
themselves are well-formed and round-trip correctly. Every module's tests
should additionally check that their own outputs validate against these
models; this file just proves the models themselves work on a clean clone.
"""

from datetime import datetime

from shared.schemas import (
    AgentDecision,
    AgentRole,
    ForecastOutput,
    HouseholdOutcome,
    HouseholdProfile,
    MarketState,
    OrderSide,
    RunSummary,
    TradeEvent,
)


def test_household_profile_roundtrip():
    profile = HouseholdProfile(
        household_id="hh_001",
        role=AgentRole.PROSUMER,
        has_solar=True,
        solar_capacity_kw=4.5,
        battery_capacity_kwh=10.0,
        grid_export_tariff_eur_per_kwh=0.08,
        grid_import_tariff_eur_per_kwh=0.32,
    )
    assert profile.model_validate(profile.model_dump()) == profile


def test_forecast_output_roundtrip():
    forecast = ForecastOutput(
        household_id="hh_001",
        tick=1,
        timestamp=datetime(2026, 1, 1, 8, 0),
        predicted_demand_kwh=1.2,
        predicted_solar_generation_kwh=2.5,
        predicted_net_position_kwh=1.3,
        confidence=0.8,
    )
    assert forecast.model_validate(forecast.model_dump()) == forecast


def test_agent_decision_requires_positive_quantity():
    import pytest

    with pytest.raises(Exception):
        AgentDecision(
            household_id="hh_001",
            tick=1,
            side=OrderSide.SELL,
            quantity_kwh=0,  # invalid — must be > 0
            limit_price_eur_per_kwh=0.15,
            strategy_name="rule_based_baseline",
        )


def test_trade_event_and_market_state():
    trade = TradeEvent(
        trade_id="t_001",
        tick=1,
        timestamp=datetime(2026, 1, 1, 8, 0),
        buyer_id="hh_002",
        seller_id="hh_001",
        quantity_kwh=1.0,
        clearing_price_eur_per_kwh=0.18,
    )
    state = MarketState(
        tick=1,
        timestamp=datetime(2026, 1, 1, 8, 0),
        trades=[trade],
        unmatched_buy_orders=[],
        unmatched_sell_orders=[],
        clearing_price_eur_per_kwh=0.18,
    )
    assert state.trades[0].trade_id == "t_001"


def test_household_outcome_and_run_summary():
    outcome = HouseholdOutcome(
        household_id="hh_001",
        total_cost_eur_p2p=45.0,
        total_cost_eur_grid_baseline=60.0,
        savings_eur=15.0,
        savings_pct=25.0,
    )
    summary = RunSummary(
        run_id="run_001",
        strategy_name="rule_based_baseline",
        num_households=10,
        num_ticks=48,
        total_savings_eur=150.0,
        avg_savings_pct=25.0,
        peak_load_reduction_pct=8.0,
        co2_avoided_kg=12.5,
    )
    assert outcome.savings_pct == 25.0
    assert summary.strategy_name == "rule_based_baseline"
