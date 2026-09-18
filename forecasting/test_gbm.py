"""Tests for the LightGBM demand forecaster.

The two properties that matter for the pipeline: it satisfies the same contract
as the baseline (so the orchestrator can swap it in blind), and it never peeks at
or beyond the tick it is predicting.
"""

from datetime import datetime, timedelta

import pytest

from data.synthetic import default_profiles, generate_household_series
from forecasting.base import Forecaster
from forecasting.gbm import MAX_LAG, GBMForecaster
from shared.schemas import ForecastOutput

EPOCH = datetime(2026, 1, 5)  # a Monday, matching the synthetic weekend rule
TICK_MINUTES = 30


def _series(days: int = 8, seed: int = 21) -> tuple[list[float], list[float]]:
    """A prosumer's demand + solar series to forecast against."""
    profile = next(p for p in default_profiles(4) if p.has_solar)
    return generate_household_series(profile, days=days, seed=seed)


def _forecast_at(forecaster: GBMForecaster, tick: int, demand, solar) -> ForecastOutput:
    """Forecast one tick from history strictly preceding it."""
    return forecaster.forecast(
        household_id="hh_000",
        tick=tick,
        timestamp=EPOCH + timedelta(minutes=TICK_MINUTES * tick),
        demand_history_kwh=demand[:tick],
        solar_history_kwh=solar[:tick],
    )


def test_satisfies_the_forecaster_protocol() -> None:
    """Drop-in for NaiveForecaster: the orchestrator swaps one constructor."""
    assert isinstance(GBMForecaster(), Forecaster)


def test_cold_start_returns_a_valid_contract_before_any_fit() -> None:
    """Tick 0 with no history still produces a usable ForecastOutput, not an error."""
    forecaster = GBMForecaster()
    result = _forecast_at(forecaster, 0, [], [])
    assert isinstance(result, ForecastOutput)
    assert result.predicted_demand_kwh >= 0.0
    assert result.predicted_solar_generation_kwh == 0.0
    assert result.confidence == 0.0


def test_fits_once_enough_history_accrues() -> None:
    """The model stays on the naive fallback until min_train_ticks, then trains."""
    demand, solar = _series()
    forecaster = GBMForecaster(min_train_ticks=192)
    _forecast_at(forecaster, 100, demand, solar)
    assert forecaster._model is None
    _forecast_at(forecaster, 200, demand, solar)
    assert forecaster._model is not None


def test_net_position_is_solar_minus_demand() -> None:
    """The field agents trade on must stay internally consistent."""
    demand, solar = _series()
    result = _forecast_at(GBMForecaster(), 250, demand, solar)
    expected = result.predicted_solar_generation_kwh - result.predicted_demand_kwh
    assert result.predicted_net_position_kwh == pytest.approx(expected)


def test_predictions_are_non_negative() -> None:
    """Negative predicted energy would be meaningless and breaks the contract's ge=0."""
    demand, solar = _series()
    forecaster = GBMForecaster()
    for tick in (10, 150, 300, 350):
        result = _forecast_at(forecaster, tick, demand, solar)
        assert result.predicted_demand_kwh >= 0.0
        assert result.predicted_solar_generation_kwh >= 0.0


def test_does_not_look_at_or_beyond_the_predicted_tick() -> None:
    """Corrupting the future must not change a prediction — the leakage guard."""
    demand, solar = _series()
    tick = 300
    clean = _forecast_at(GBMForecaster(), tick, demand, solar)

    poisoned_demand = list(demand)
    poisoned_solar = list(solar)
    for index in range(tick, len(poisoned_demand)):
        poisoned_demand[index] = 99.0
        poisoned_solar[index] = 99.0
    poisoned = _forecast_at(GBMForecaster(), tick, poisoned_demand, poisoned_solar)

    assert poisoned.predicted_demand_kwh == pytest.approx(clean.predicted_demand_kwh)
    assert poisoned.predicted_solar_generation_kwh == pytest.approx(
        clean.predicted_solar_generation_kwh
    )


def test_beats_the_naive_fallback_on_a_seasonal_series() -> None:
    """A fitted GBM tracks the daily shape better than the cold-start rolling mean."""
    demand, solar = _series(days=10)
    forecaster = GBMForecaster()
    fitted_error = 0.0
    fallback_error = 0.0
    for tick in range(384, 432):  # a full day, well after the model has trained
        result = _forecast_at(forecaster, tick, demand, solar)
        fitted_error += abs(result.predicted_demand_kwh - demand[tick])
        fallback_error += abs(sum(demand[tick - 4 : tick]) / 4 - demand[tick])
    assert fitted_error < fallback_error


def test_solar_climatology_uses_same_time_of_day() -> None:
    """Solar is predicted from the same half-hour on previous days, not a flat mean."""
    demand, solar = _series()
    forecaster = GBMForecaster()
    midday = _forecast_at(forecaster, 4 * 48 + 26, demand, solar)
    midnight = _forecast_at(forecaster, 4 * 48 + 2, demand, solar)
    assert midday.predicted_solar_generation_kwh > 0.0
    assert midnight.predicted_solar_generation_kwh == 0.0


def test_rejects_a_training_threshold_below_the_longest_lag() -> None:
    """Fitting with fewer ticks than the longest lag would build empty feature rows."""
    with pytest.raises(ValueError):
        GBMForecaster(min_train_ticks=MAX_LAG)
