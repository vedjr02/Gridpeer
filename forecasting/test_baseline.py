"""Tests for the naive baseline forecaster.

Verifies the day-1 skeleton produces a contract-valid ForecastOutput and that
its rolling-average / fallback logic behaves as intended.
"""

from datetime import datetime

import pytest

from forecasting import NaiveForecaster
from shared.schemas import ForecastOutput


def test_forecast_returns_valid_contract() -> None:
    """Output is a ForecastOutput satisfying the shared contract."""
    result = NaiveForecaster().forecast(
        household_id="h1",
        tick=10,
        timestamp=datetime(2026, 9, 16, 8, 0),
        demand_history_kwh=[1.0, 2.0, 3.0, 4.0],
        solar_history_kwh=[0.0, 1.0, 2.0, 3.0],
    )
    assert isinstance(result, ForecastOutput)
    assert result.household_id == "h1"
    assert result.tick == 10


def test_rolling_mean_over_window() -> None:
    """Prediction equals the mean of the last `window` observations."""
    result = NaiveForecaster(window=2).forecast(
        household_id="h1",
        tick=1,
        timestamp=datetime(2026, 9, 16, 8, 0),
        demand_history_kwh=[10.0, 2.0, 4.0],  # last 2 -> mean 3.0
        solar_history_kwh=[0.0, 1.0, 5.0],  # last 2 -> mean 3.0
    )
    assert result.predicted_demand_kwh == 3.0
    assert result.predicted_solar_generation_kwh == 3.0
    assert result.predicted_net_position_kwh == 0.0


def test_empty_history_uses_flat_fallback() -> None:
    """No history -> flat demand fallback and zero solar; deficit net position."""
    result = NaiveForecaster(flat_demand_kwh=0.5).forecast(
        household_id="h1",
        tick=0,
        timestamp=datetime(2026, 9, 16, 0, 0),
        demand_history_kwh=[],
        solar_history_kwh=[],
    )
    assert result.predicted_demand_kwh == 0.5
    assert result.predicted_solar_generation_kwh == 0.0
    assert result.predicted_net_position_kwh == -0.5
    assert result.confidence == 0.0


def test_confidence_is_high_on_a_predictable_series() -> None:
    """A flat series is predicted exactly, so recent error is ~0 and confidence saturates."""
    forecaster = NaiveForecaster(window=4)
    result = forecaster.forecast(
        "h1", 12, datetime(2026, 9, 16, 8, 0), [2.0] * 12, [0.0] * 12
    )
    assert result.confidence == pytest.approx(1.0)


def test_confidence_drops_on_an_erratic_series() -> None:
    """A series the rolling average keeps missing scores lower than one it tracks."""
    forecaster = NaiveForecaster(window=4)
    ts = datetime(2026, 9, 16, 8, 0)
    steady = forecaster.forecast("h1", 12, ts, [2.0] * 12, [0.0] * 12)
    erratic = forecaster.forecast(
        "h1", 12, ts, [0.1, 5.0, 0.2, 6.0, 0.1, 5.5, 0.3, 6.5, 0.2, 5.0, 0.1, 6.0], [0.0] * 12
    )
    assert erratic.confidence < steady.confidence


def test_confidence_respects_the_contract_bounds() -> None:
    """Confidence always lands in [0, 1], whatever the history looks like."""
    forecaster = NaiveForecaster(window=4)
    ts = datetime(2026, 9, 16, 8, 0)
    for demand in ([], [1.0], [0.0] * 6, [0.1, 40.0] * 6):
        result = forecaster.forecast("h1", len(demand), ts, demand, [0.0] * len(demand))
        assert 0.0 <= result.confidence <= 1.0
