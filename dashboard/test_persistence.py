"""Tests for the SQLite RunStore: every shared model round-trips, keyed by run_id."""

from datetime import datetime

import pytest

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

TIMESTAMP = datetime(2026, 1, 1, 8, 0)


@pytest.fixture
def store() -> RunStore:
    """An in-memory store, closed when the test finishes."""
    with RunStore() as open_store:
        yield open_store


def _profile(household_id: str = "hh_000") -> HouseholdProfile:
    return HouseholdProfile(
        household_id=household_id,
        role=AgentRole.PROSUMER,
        has_solar=True,
        solar_capacity_kw=3.0,
        battery_capacity_kwh=5.0,
        grid_export_tariff_eur_per_kwh=0.07,
        grid_import_tariff_eur_per_kwh=0.25,
    )


def _decision(household_id: str, side: OrderSide, tick: int = 0) -> AgentDecision:
    return AgentDecision(
        household_id=household_id,
        tick=tick,
        side=side,
        quantity_kwh=1.5,
        limit_price_eur_per_kwh=0.16,
        strategy_name="rule_based_baseline",
    )


def _trade(trade_id: str = "t_0000_000", tick: int = 0) -> TradeEvent:
    return TradeEvent(
        trade_id=trade_id,
        tick=tick,
        timestamp=TIMESTAMP,
        buyer_id="hh_001",
        seller_id="hh_000",
        quantity_kwh=1.0,
        clearing_price_eur_per_kwh=0.16,
    )


def test_household_profiles_roundtrip(store: RunStore) -> None:
    profiles = [_profile("hh_000"), _profile("hh_001")]
    store.save_household_profiles("run_a", profiles)
    assert store.load_household_profiles("run_a") == profiles


def test_forecasts_roundtrip(store: RunStore) -> None:
    forecasts = [
        ForecastOutput(
            household_id="hh_000",
            tick=tick,
            timestamp=TIMESTAMP,
            predicted_demand_kwh=1.2,
            predicted_solar_generation_kwh=2.5,
            predicted_net_position_kwh=1.3,
            confidence=0.75,
        )
        for tick in range(3)
    ]
    store.save_forecasts("run_a", forecasts)
    assert store.load_forecasts("run_a") == forecasts


def test_decisions_roundtrip_preserving_enum(store: RunStore) -> None:
    """OrderSide survives the round trip as an enum, not a bare string."""
    decisions = [_decision("hh_000", OrderSide.SELL), _decision("hh_001", OrderSide.BUY)]
    store.save_decisions("run_a", decisions)
    loaded = store.load_decisions("run_a")
    assert loaded == decisions
    assert loaded[0].side is OrderSide.SELL


def test_trades_roundtrip_and_filter_by_tick(store: RunStore) -> None:
    trades = [_trade("t_0000_000", tick=0), _trade("t_0001_000", tick=1)]
    store.save_trades("run_a", trades)
    assert store.load_trades("run_a") == trades
    assert store.load_trades("run_a", tick=1) == [trades[1]]


def test_market_state_roundtrip_rebuilds_trades(store: RunStore) -> None:
    """Trades live only in the trades table, but come back attached to the state."""
    state = MarketState(
        tick=0,
        timestamp=TIMESTAMP,
        trades=[_trade()],
        unmatched_buy_orders=[_decision("hh_002", OrderSide.BUY)],
        unmatched_sell_orders=[_decision("hh_003", OrderSide.SELL)],
        clearing_price_eur_per_kwh=0.16,
    )
    store.save_market_state("run_a", state)
    assert store.load_market_states("run_a") == [state]


def test_market_state_with_no_trades_roundtrips(store: RunStore) -> None:
    """A tick that cleared nothing keeps its None clearing price."""
    state = MarketState(
        tick=7,
        timestamp=TIMESTAMP,
        trades=[],
        unmatched_buy_orders=[],
        unmatched_sell_orders=[],
        clearing_price_eur_per_kwh=None,
    )
    store.save_market_state("run_a", state)
    loaded = store.load_market_states("run_a")
    assert loaded == [state]
    assert loaded[0].clearing_price_eur_per_kwh is None


def test_outcomes_and_summary_roundtrip(store: RunStore) -> None:
    outcomes = [
        HouseholdOutcome(
            household_id="hh_000",
            total_cost_eur_p2p=45.0,
            total_cost_eur_grid_baseline=60.0,
            savings_eur=15.0,
            savings_pct=25.0,
        )
    ]
    summary = RunSummary(
        run_id="run_a",
        strategy_name="rule_based_baseline",
        num_households=1,
        num_ticks=48,
        total_savings_eur=15.0,
        avg_savings_pct=25.0,
        peak_load_reduction_pct=8.0,
        co2_avoided_kg=12.5,
    )
    store.save_household_outcomes("run_a", outcomes)
    store.save_run_summary(summary)
    assert store.load_household_outcomes("run_a") == outcomes
    assert store.load_run_summary("run_a") == summary


def test_runs_are_isolated_by_run_id(store: RunStore) -> None:
    """Two runs in one database do not see each other's rows."""
    store.save_trades("run_a", [_trade("t_0000_000")])
    store.save_trades("run_b", [_trade("t_0000_000"), _trade("t_0000_001")])
    assert len(store.load_trades("run_a")) == 1
    assert len(store.load_trades("run_b")) == 2


def test_writes_are_idempotent(store: RunStore) -> None:
    """Re-saving a tick overwrites it rather than doubling the history."""
    store.save_trades("run_a", [_trade()])
    store.save_trades("run_a", [_trade()])
    assert len(store.load_trades("run_a")) == 1


def test_load_missing_summary_returns_none(store: RunStore) -> None:
    assert store.load_run_summary("never_ran") is None


def test_list_runs_only_lists_summarised_runs(store: RunStore) -> None:
    store.save_trades("run_unsummarised", [_trade()])
    store.save_run_summary(
        RunSummary(
            run_id="run_a",
            strategy_name="rule_based_baseline",
            num_households=1,
            num_ticks=1,
            total_savings_eur=0.0,
            avg_savings_pct=0.0,
            peak_load_reduction_pct=0.0,
            co2_avoided_kg=0.0,
        )
    )
    assert store.list_runs() == ["run_a"]


def test_delete_run_removes_every_table(store: RunStore) -> None:
    store.save_household_profiles("run_a", [_profile()])
    store.save_trades("run_a", [_trade()])
    store.save_decisions("run_a", [_decision("hh_000", OrderSide.SELL)])
    store.delete_run("run_a")
    assert store.load_household_profiles("run_a") == []
    assert store.load_trades("run_a") == []
    assert store.load_decisions("run_a") == []


def test_persists_to_a_file_across_connections(tmp_path) -> None:
    """A run written to disk is readable by a fresh RunStore — the point of the module."""
    db_path = tmp_path / "gridpeer.db"
    with RunStore(db_path) as writer:
        writer.save_trades("run_a", [_trade()])
    with RunStore(db_path) as reader:
        assert reader.load_trades("run_a") == [_trade()]
