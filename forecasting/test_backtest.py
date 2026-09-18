"""Tests for the hold-out backtest harness.

The harness produces a number that goes in the writeup, so the things worth
pinning down are that it evaluates the right window, gives each household its own
model instance, and never lets a model see the tick it is being scored on.
"""

from datetime import datetime, timedelta

import pytest

from data.synthetic import ticks_per_day
from forecasting.backtest import (
    BacktestResult,
    SeriesErrors,
    backtest_models,
    format_table,
    rolling_predictions,
)
from forecasting.baseline import NaiveForecaster
from shared.schemas import ForecastOutput


class _SpyForecaster:
    """Records the history lengths it was handed, so leakage is directly observable."""

    def __init__(self) -> None:
        self.seen_history_lengths: list[int] = []

    def forecast(
        self,
        household_id: str,
        tick: int,
        timestamp: datetime,
        demand_history_kwh,
        solar_history_kwh,
    ) -> ForecastOutput:
        """Return a constant forecast while logging how much history it received."""
        self.seen_history_lengths.append(len(demand_history_kwh))
        return ForecastOutput(
            household_id=household_id,
            tick=tick,
            timestamp=timestamp,
            predicted_demand_kwh=1.0,
            predicted_solar_generation_kwh=0.0,
            predicted_net_position_kwh=-1.0,
            confidence=0.5,
        )


def test_rolling_predictions_cover_exactly_the_holdout() -> None:
    """One prediction per test tick, each from history of exactly that tick's length."""
    spy = _SpyForecaster()
    demand = [1.0] * 20
    solar = [0.0] * 20
    demand_pred, solar_pred = rolling_predictions(spy, "hh_000", demand, solar, first_test_tick=15)
    assert len(demand_pred) == len(solar_pred) == 5
    assert spy.seen_history_lengths == [15, 16, 17, 18, 19]


def test_rolling_predictions_advance_the_timestamp_per_tick() -> None:
    """Calendar features depend on the timestamp, so it must track the tick index."""
    epoch = datetime(2026, 1, 5)
    seen: list[datetime] = []

    class _TimestampSpy(_SpyForecaster):
        def forecast(self, household_id, tick, timestamp, demand_history_kwh, solar_history_kwh):
            seen.append(timestamp)
            return super().forecast(
                household_id, tick, timestamp, demand_history_kwh, solar_history_kwh
            )

    rolling_predictions(
        _TimestampSpy(), "hh_000", [1.0] * 5, [0.0] * 5, first_test_tick=3, epoch=epoch
    )
    assert seen == [epoch + timedelta(minutes=30 * t) for t in (3, 4)]


def test_backtest_reports_each_household_plus_a_pooled_row() -> None:
    """Per-household rows for debugging, one ALL row for the headline number."""
    results = backtest_models({"naive": NaiveForecaster}, num_households=3, days=4, holdout_days=1)
    assert len(results) == 4
    assert [r.household_id for r in results][-1] == "ALL"
    assert all(r.model_name == "naive" for r in results)


def test_pooled_row_covers_every_household_tick() -> None:
    """The ALL row aggregates raw errors, not an average of averages."""
    holdout_days, num_households = 1, 3
    results = backtest_models(
        {"naive": NaiveForecaster}, num_households=num_households, days=4, holdout_days=holdout_days
    )
    pooled = results[-1]
    assert pooled.num_test_ticks == num_households * holdout_days * ticks_per_day()


def test_each_household_gets_its_own_forecaster_instance() -> None:
    """A fitted model carries state; sharing one across households would leak between them."""
    built: list[_SpyForecaster] = []

    def factory() -> _SpyForecaster:
        built.append(_SpyForecaster())
        return built[-1]

    backtest_models({"spy": factory}, num_households=3, days=4, holdout_days=1)
    assert len(built) == 3


def test_solar_mape_is_undefined_for_a_household_without_panels() -> None:
    """An all-zero solar series has no meaningful percentage error — reported as None."""
    results = backtest_models({"naive": NaiveForecaster}, num_households=2, days=4, holdout_days=1)
    consumer = results[1]  # default_profiles() makes index 1 the pure consumer
    assert consumer.solar.mape_pct is None
    assert consumer.demand.mape_pct is not None


def test_holdout_must_fit_inside_the_generated_series() -> None:
    """Asking to hold out everything would leave no training history at all."""
    with pytest.raises(ValueError):
        backtest_models({"naive": NaiveForecaster}, days=3, holdout_days=3)


def test_format_table_renders_a_row_per_result_and_dashes_undefined_metrics() -> None:
    """The table is a writeup artefact; undefined cells show a dash, not a crash."""
    errors = SeriesErrors(mae_kwh=0.1, normalised_mae=0.2, mape_pct=None)
    table = format_table(
        [BacktestResult("naive", "hh_000", 48, errors, errors)],
    )
    lines = table.splitlines()
    assert len(lines) == 3  # header, rule, one row
    assert "hh_000" in lines[2]
    assert "—" in lines[2]
