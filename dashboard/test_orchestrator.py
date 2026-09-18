"""Tests for the orchestration loop: the forecasting stage, and the full pipeline."""

from datetime import datetime
from pathlib import Path

import pytest

from dashboard import orchestrator
from dashboard.orchestrator import (
    HouseholdSeries,
    _grid_imports_kwh,
    _placeholder_series,
    demo_households,
    run_forecasts,
    run_pipeline,
)
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
from simulation.market import clear_tick

_TS = datetime(2026, 1, 1, 0, 0)


# -- forecasting stage ------------------------------------------------------

def test_demo_households_shape() -> None:
    """Demo households carry equal-length demand and solar series."""
    households = demo_households(num_households=3, days=1)
    assert len(households) == 3
    for h in households:
        assert len(h.demand_kwh) == len(h.solar_kwh) == 48  # 1 day, half-hourly


def test_run_emits_valid_forecastoutput_per_household_per_tick() -> None:
    """Stream length == households * ticks, each a contract-valid ForecastOutput."""
    households = demo_households(num_households=2, days=1)
    stream = run_forecasts(households, num_ticks=10)
    assert len(stream) == 2 * 10
    assert all(isinstance(f, ForecastOutput) for f in stream)
    assert {f.household_id for f in stream} == {"hh_000", "hh_001"}
    assert max(f.tick for f in stream) == 9


def test_net_position_is_solar_minus_demand() -> None:
    """The contract's net position stays internally consistent."""
    stream = run_forecasts(demo_households(num_households=1, days=1), num_ticks=5)
    for f in stream:
        assert f.predicted_net_position_kwh == pytest.approx(
            f.predicted_solar_generation_kwh - f.predicted_demand_kwh
        )


def test_num_ticks_capped_to_series_length() -> None:
    """Asking for more ticks than exist is capped, not an error."""
    households = demo_households(num_households=1, days=1)  # 48 ticks
    stream = run_forecasts(households, num_ticks=1000)
    assert max(f.tick for f in stream) == 47


def test_empty_households_raises() -> None:
    with pytest.raises(ValueError):
        run_forecasts([], num_ticks=5)


def test_forecasts_never_see_the_tick_being_predicted() -> None:
    """Tick 0 has no history at all, so it falls back rather than peeking ahead."""
    households = demo_households(num_households=1, days=1)
    stream = run_forecasts(households, num_ticks=1)
    assert stream[0].confidence == 0.0


# -- full pipeline ----------------------------------------------------------

def test_pipeline_produces_every_stage_of_the_contract() -> None:
    """Forecast -> decide -> clear -> roll up, all as shared models."""
    households = demo_households(num_households=4, days=1)
    result = run_pipeline(households, num_ticks=24, run_id="run_test")

    assert result.run_id == "run_test"
    assert len(result.forecasts) == 4 * 24
    assert all(isinstance(f, ForecastOutput) for f in result.forecasts)
    assert all(isinstance(d, AgentDecision) for d in result.decisions)
    assert len(result.market_states) == 24
    assert all(isinstance(s, MarketState) for s in result.market_states)
    assert all(isinstance(t, TradeEvent) for t in result.trades)
    assert len(result.outcomes) == 4
    assert all(isinstance(o, HouseholdOutcome) for o in result.outcomes)
    assert isinstance(result.summary, RunSummary)


def test_pipeline_actually_clears_trades() -> None:
    """A mixed book must trade — a pipeline that never matches proves nothing."""
    result = run_pipeline(demo_households(num_households=4, days=2))
    assert result.trades, "no trades cleared; the demo book is not crossing"


def test_pipeline_ticks_are_contiguous_from_zero() -> None:
    result = run_pipeline(demo_households(num_households=2, days=1), num_ticks=12)
    assert [s.tick for s in result.market_states] == list(range(12))


def test_one_decision_per_household_per_tick_at_most() -> None:
    """The market contract requires it, so the loop must not submit duplicates."""
    result = run_pipeline(demo_households(num_households=4, days=1), num_ticks=20)
    seen = [(d.household_id, d.tick) for d in result.decisions]
    assert len(seen) == len(set(seen))


def test_summary_counts_match_the_run() -> None:
    result = run_pipeline(demo_households(num_households=3, days=1), num_ticks=15)
    assert result.summary is not None
    assert result.summary.num_households == 3
    assert result.summary.num_ticks == 15
    assert result.summary.strategy_name == "rule_based_baseline"


def test_summary_totals_agree_with_the_per_household_outcomes() -> None:
    """The headline number is exactly the sum of its parts — no rounding drift."""
    result = run_pipeline(demo_households(num_households=4, days=1))
    assert result.summary is not None
    assert result.summary.total_savings_eur == pytest.approx(
        sum(o.savings_eur for o in result.outcomes)
    )


def test_co2_avoided_tracks_cleared_volume() -> None:
    """CO2 avoided is traded kWh at the grid factor and nothing else."""
    result = run_pipeline(demo_households(num_households=4, days=1), grid_co2_g_per_kwh=500.0)
    traded_kwh = sum(t.quantity_kwh for t in result.trades)
    assert result.summary is not None
    assert result.summary.co2_avoided_kg == pytest.approx(traded_kwh * 0.5)


def test_pipeline_is_deterministic() -> None:
    """Same households, same run: same savings. Reproducibility is the debugging tool."""
    households = demo_households(num_households=4, days=1)
    first = run_pipeline(households, num_ticks=20, run_id="run_a")
    second = run_pipeline(households, num_ticks=20, run_id="run_a")
    assert first.summary is not None and second.summary is not None
    assert first.summary.total_savings_eur == pytest.approx(second.summary.total_savings_eur)
    assert len(first.trades) == len(second.trades)


def test_pipeline_with_a_single_household_clears_nothing_and_saves_nothing() -> None:
    """Nobody to trade with means honest zeros, not a crash or an invented saving."""
    result = run_pipeline(demo_households(num_households=1, days=1), num_ticks=10)
    assert result.trades == []
    assert result.summary is not None
    assert result.summary.total_savings_eur == pytest.approx(0.0)
    assert result.summary.co2_avoided_kg == pytest.approx(0.0)


def test_pipeline_rejects_an_empty_run() -> None:
    with pytest.raises(ValueError):
        run_pipeline([], num_ticks=5)


def test_battery_changes_the_result_and_still_satisfies_the_contract() -> None:
    """The battery flag is wired through to simulation's environment, not ignored."""
    households = demo_households(num_households=4, days=1)
    without = run_pipeline(households, num_ticks=48, use_battery=False)
    with_battery = run_pipeline(households, num_ticks=48, use_battery=True)
    assert isinstance(with_battery.summary, RunSummary)
    assert with_battery.summary.total_savings_eur != pytest.approx(
        without.summary.total_savings_eur  # type: ignore[union-attr]
    )


# -- persistence wiring -----------------------------------------------------

def test_pipeline_persists_every_stage_to_the_store() -> None:
    """What went through the loop is what comes back out of the database."""
    with RunStore() as store:
        result = run_pipeline(
            demo_households(num_households=4, days=1),
            num_ticks=20,
            run_id="run_persisted",
            store=store,
        )
        assert store.load_forecasts("run_persisted") == sorted(
            result.forecasts, key=lambda f: (f.tick, f.household_id)
        )
        assert store.load_decisions("run_persisted") == sorted(
            result.decisions, key=lambda d: (d.tick, d.household_id)
        )
        assert store.load_market_states("run_persisted") == result.market_states
        assert store.load_household_outcomes("run_persisted") == result.outcomes
        assert store.load_run_summary("run_persisted") == result.summary
        assert store.load_household_profiles("run_persisted") == [
            h.profile for h in demo_households(num_households=4, days=1)
        ]
        assert store.list_runs() == ["run_persisted"]


def test_pipeline_opens_and_closes_its_own_store_from_db_path(tmp_path) -> None:
    """Passing db_path instead of a store leaves a readable file behind."""
    db_path = tmp_path / "run.db"
    result = run_pipeline(
        demo_households(num_households=3, days=1),
        num_ticks=10,
        run_id="run_file",
        db_path=db_path,
    )
    with RunStore(db_path) as store:
        assert store.load_run_summary("run_file") == result.summary
        assert len(store.load_market_states("run_file")) == 10


def test_two_runs_share_one_database_without_colliding(tmp_path) -> None:
    """Baseline versus a learned policy is the comparison this store exists for."""
    households = demo_households(num_households=3, days=1)
    with RunStore(tmp_path / "runs.db") as store:
        run_pipeline(households, num_ticks=10, run_id="run_a", store=store)
        run_pipeline(households, num_ticks=20, run_id="run_b", store=store)
        assert sorted(store.list_runs()) == ["run_a", "run_b"]
        assert len(store.load_market_states("run_a")) == 10
        assert len(store.load_market_states("run_b")) == 20


# -- grid-import accounting -------------------------------------------------

def test_grid_imports_count_deficits_only() -> None:
    """Exporting surplus is not load, so only deficits reach the import figures."""
    state = clear_tick(tick=0, timestamp=_TS, orders=[])
    p2p_kwh, grid_only_kwh = _grid_imports_kwh(state, {"a": -2.0, "b": 3.0})
    assert grid_only_kwh == pytest.approx(2.0)
    assert p2p_kwh == pytest.approx(2.0)  # nothing cleared, so P2P draws the same


def test_a_cleared_trade_reduces_the_p2p_import_not_the_baseline() -> None:
    """The whole peak-load claim rests on this: matched energy never touches the grid."""
    orders = [
        AgentDecision(
            household_id="seller",
            tick=0,
            side=OrderSide.SELL,
            quantity_kwh=2.0,
            limit_price_eur_per_kwh=0.10,
            strategy_name="test",
        ),
        AgentDecision(
            household_id="buyer",
            tick=0,
            side=OrderSide.BUY,
            quantity_kwh=2.0,
            limit_price_eur_per_kwh=0.20,
            strategy_name="test",
        ),
    ]
    state = clear_tick(tick=0, timestamp=_TS, orders=orders)
    assert state.trades, "book should cross"
    p2p_kwh, grid_only_kwh = _grid_imports_kwh(state, {"seller": 2.0, "buyer": -2.0})
    assert grid_only_kwh == pytest.approx(2.0)
    assert p2p_kwh == pytest.approx(0.0)


# -- synthetic data fallback ------------------------------------------------

def test_placeholder_series_are_the_right_length_and_non_negative() -> None:
    """The local stand-in for data.synthetic must satisfy the same basic shape."""
    demand, solar = _placeholder_series(AgentRole.PROSUMER, solar_capacity_kw=3.0, days=2, seed=0)
    assert len(demand) == len(solar) == 96
    assert all(value >= 0 for value in demand)
    assert all(value >= 0 for value in solar)
    assert max(solar) > 0, "a household with panels should generate something"


def test_demo_households_series_feed_the_pipeline_unchanged() -> None:
    """Whatever the series source, HouseholdSeries is what the loop consumes."""
    households = demo_households(num_households=2, days=1)
    assert all(isinstance(h, HouseholdSeries) for h in households)
    assert all(len(h.demand_kwh) == len(h.solar_kwh) for h in households)


# -- pluggable strategy + persisted default run -----------------------------

class _AlwaysBuy:
    """Stand-in for a learned policy: satisfies TradingStrategy, shares no base class."""

    def decide(
        self, forecast: ForecastOutput, profile: HouseholdProfile
    ) -> AgentDecision | None:
        """Buy 1 kWh every tick at the household's import tariff."""
        return AgentDecision(
            household_id=forecast.household_id,
            tick=forecast.tick,
            side=OrderSide.BUY,
            quantity_kwh=1.0,
            limit_price_eur_per_kwh=profile.grid_import_tariff_eur_per_kwh,
            strategy_name="always_buy",
        )


def test_run_pipeline_accepts_any_strategy_with_decide() -> None:
    """An RL policy plugs in by exposing decide — no subclassing the baseline."""
    result = run_pipeline(
        demo_households(num_households=2, days=1),
        num_ticks=10,
        strategy=_AlwaysBuy(),
        strategy_name="always_buy",
    )
    assert len(result.decisions) == 20
    assert all(d.strategy_name == "always_buy" for d in result.decisions)
    assert result.summary is not None
    assert result.summary.strategy_name == "always_buy"


def test_main_persists_a_run_the_dashboard_can_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`make run-sim` must leave a run behind — the dashboard renders stored runs only."""
    db_path = tmp_path / "gridpeer.db"
    monkeypatch.setattr(orchestrator, "DEFAULT_DB_PATH", db_path)
    orchestrator.main()

    with RunStore(db_path) as store:
        run_ids = store.list_runs()
        assert len(run_ids) == 1
        summary = store.load_run_summary(run_ids[0])
        assert summary is not None
        assert summary.num_households == 4
        assert store.load_household_outcomes(run_ids[0])
        assert store.load_market_states(run_ids[0])
