"""Tests for the walk-forward backtest runner."""

import pytest

from forecasting.backtest import BacktestResult, backtest_household
from forecasting.baseline import NaiveForecaster


def test_perfect_on_constant_series() -> None:
    """A flat series is predicted exactly by a rolling average: zero error."""
    demand = [2.0] * 10
    solar = [1.0] * 10
    result = backtest_household(NaiveForecaster(window=3), "h1", demand, solar)
    assert isinstance(result, BacktestResult)
    assert result.demand_mae == 0.0
    assert result.demand_mape == 0.0
    assert result.solar_mae == 0.0


def test_scores_all_ticks_after_warmup() -> None:
    """n_points equals series length minus the warm-up (min_history)."""
    demand = [1.0, 2.0, 3.0, 4.0, 5.0]
    solar = [0.0, 0.0, 1.0, 1.0, 2.0]
    result = backtest_household(NaiveForecaster(), "h1", demand, solar, min_history=2)
    assert result.n_points == 3


def test_nonzero_error_on_varying_series() -> None:
    """A trending series is not predicted perfectly by a lagging average."""
    demand = [float(i) for i in range(10)]
    solar = [0.0] * 10
    result = backtest_household(NaiveForecaster(window=2), "h1", demand, solar)
    assert result.demand_mae > 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        backtest_household(NaiveForecaster(), "h1", [1.0, 2.0], [1.0])


def test_series_too_short_raises() -> None:
    with pytest.raises(ValueError):
        backtest_household(NaiveForecaster(), "h1", [1.0], [1.0], min_history=1)
