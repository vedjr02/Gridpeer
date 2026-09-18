"""
Cross-module contract tests.

Two halves, in this order:

1. **The models themselves.** Each shared contract is well-formed and round-trips
   through ``model_dump``/``model_validate`` on a clean clone, independently of any
   module that uses it.
2. **The pipeline speaks them.** One real end-to-end run supplies every object under
   test, and each stage's output is validated against the model the next stage
   consumes. Nothing is hand-typed from one stage into the next — that is the
   week-1 definition of "done" in the root AGENTS.md, and this is where it is
   actually checked rather than assumed.

Individual modules still test their own logic in their own test files; this file
only ever asserts against ``shared/schemas.py``.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from dashboard.orchestrator import demo_households, run_pipeline
from dashboard.persistence import RunStore
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
    with pytest.raises(ValidationError):
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


# ---------------------------------------------------------------------------
# End-to-end: every stage's output validates against the shared contracts.
#
# The tests above prove the models are well-formed on a clean clone. These prove
# the pipeline actually speaks them — that what forecasting/ emits is what agents/
# consumes, what agents/ emits is what simulation/ clears, and what simulation/
# emits is what the dashboard rolls up. Nothing here is hand-typed from one stage
# into the next; a single real run supplies every object under test, which is the
# week-1 definition of "done" in the root AGENTS.md.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_run():
    """One real pipeline run, shared by every end-to-end contract test below."""
    return run_pipeline(
        demo_households(num_households=4, days=1), num_ticks=24, run_id="run_contracts"
    )


def _revalidates(model):
    """True if a model survives a dump/validate round trip unchanged."""
    return type(model).model_validate(model.model_dump()) == model


def test_forecast_stage_output_satisfies_the_contract(pipeline_run):
    assert pipeline_run.forecasts
    for forecast in pipeline_run.forecasts:
        assert isinstance(forecast, ForecastOutput)
        assert _revalidates(forecast)
        assert forecast.predicted_net_position_kwh == pytest.approx(
            forecast.predicted_solar_generation_kwh - forecast.predicted_demand_kwh
        )


def test_agent_stage_output_satisfies_the_contract(pipeline_run):
    assert pipeline_run.decisions
    for decision in pipeline_run.decisions:
        assert isinstance(decision, AgentDecision)
        assert _revalidates(decision)
        assert decision.quantity_kwh > 0
        assert decision.side in (OrderSide.BUY, OrderSide.SELL)
        assert decision.strategy_name


def test_market_stage_output_satisfies_the_contract(pipeline_run):
    assert pipeline_run.market_states
    for state in pipeline_run.market_states:
        assert isinstance(state, MarketState)
        assert _revalidates(state)
        for trade in state.trades:
            assert isinstance(trade, TradeEvent)
            assert trade.tick == state.tick
            assert trade.quantity_kwh > 0
            assert trade.buyer_id != trade.seller_id
        if not state.trades:
            assert state.clearing_price_eur_per_kwh is None


def test_rollup_stage_output_satisfies_the_contract(pipeline_run):
    assert pipeline_run.outcomes
    for outcome in pipeline_run.outcomes:
        assert isinstance(outcome, HouseholdOutcome)
        assert _revalidates(outcome)
        assert outcome.savings_eur == pytest.approx(
            outcome.total_cost_eur_grid_baseline - outcome.total_cost_eur_p2p
        )

    summary = pipeline_run.summary
    assert isinstance(summary, RunSummary)
    assert _revalidates(summary)
    assert summary.num_households == len(pipeline_run.outcomes)


def test_stages_reference_the_same_households_and_ticks(pipeline_run):
    """The stages are wired to each other, not each to its own private set of ids."""
    households = {f.household_id for f in pipeline_run.forecasts}
    assert {d.household_id for d in pipeline_run.decisions} <= households
    assert {o.household_id for o in pipeline_run.outcomes} == households

    ticks = {f.tick for f in pipeline_run.forecasts}
    assert {s.tick for s in pipeline_run.market_states} == ticks
    assert {t.tick for t in pipeline_run.trades} <= ticks


def test_every_traded_counterparty_submitted_a_matching_order(pipeline_run):
    """A trade can only exist between households that actually bid and asked."""
    buyers = {(d.tick, d.household_id) for d in pipeline_run.decisions if d.side is OrderSide.BUY}
    sellers = {
        (d.tick, d.household_id) for d in pipeline_run.decisions if d.side is OrderSide.SELL
    }
    for trade in pipeline_run.trades:
        assert (trade.tick, trade.buyer_id) in buyers
        assert (trade.tick, trade.seller_id) in sellers


def test_contracts_survive_the_database_round_trip(pipeline_run):
    """Persistence must not be a lossy stage — what goes in is what comes out."""
    with RunStore() as store:
        run_id = pipeline_run.run_id
        store.save_household_profiles(
            run_id, [h.profile for h in demo_households(num_households=4, days=1)]
        )
        store.save_forecasts(run_id, pipeline_run.forecasts)
        store.save_decisions(run_id, pipeline_run.decisions)
        for state in pipeline_run.market_states:
            store.save_market_state(run_id, state)
        store.save_household_outcomes(run_id, pipeline_run.outcomes)
        store.save_run_summary(pipeline_run.summary)

        assert store.load_market_states(run_id) == pipeline_run.market_states
        assert store.load_household_outcomes(run_id) == pipeline_run.outcomes
        assert store.load_run_summary(run_id) == pipeline_run.summary
        assert len(store.load_forecasts(run_id)) == len(pipeline_run.forecasts)
        assert len(store.load_decisions(run_id)) == len(pipeline_run.decisions)
