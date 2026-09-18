"""Tests for the shared error metrics and the confidence score.

The confidence field is the one number agents may use to bid conservatively, so
its edge cases — no history, a perfectly predictable series, an all-zero solar
series overnight — matter more than its typical value.
"""

import pytest

from forecasting.metrics import (
    ERROR_SCALE_FLOOR_KWH,
    mae,
    mape,
    normalised_mae,
    recent_error_confidence,
)


def _last_value(history: list[float]) -> float:
    """Trivial predictor: next tick equals the previous one."""
    return history[-1] if history else 0.0


def test_mae_is_mean_absolute_difference() -> None:
    """MAE averages the absolute gaps, in kWh."""
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert mae([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.5)


def test_mae_rejects_mismatched_or_empty_series() -> None:
    """Silently comparing series of different lengths would invalidate the backtest."""
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        mae([], [])


def test_mape_skips_near_zero_actuals() -> None:
    """Actuals at ~0 are excluded rather than producing an infinite percentage."""
    result = mape([1.0, 0.0], [2.0, 0.5])
    assert result == pytest.approx(100.0)


def test_mape_undefined_when_every_actual_is_near_zero() -> None:
    """An all-dark solar window has no meaningful MAPE — callers must handle that."""
    with pytest.raises(ValueError):
        mape([0.0, 0.0], [0.1, 0.1])


def test_normalised_mae_is_safe_on_all_zero_series() -> None:
    """nMAE stays finite where MAPE cannot, using a floor on the denominator."""
    assert normalised_mae([0.0, 0.0], [0.0, 0.0]) == 0.0
    expected = 0.1 / ERROR_SCALE_FLOOR_KWH
    assert normalised_mae([0.0, 0.0], [0.1, 0.1]) == pytest.approx(expected)


def test_confidence_is_zero_without_enough_history() -> None:
    """A model with no track record reports no confidence."""
    assert recent_error_confidence([], _last_value, lookback=4, warmup_ticks=4) == 0.0
    assert recent_error_confidence([1.0], _last_value, lookback=4, warmup_ticks=4) == 0.0


def test_confidence_is_high_on_a_perfectly_predictable_series() -> None:
    """A constant series is predicted exactly, so confidence saturates."""
    confidence = recent_error_confidence([2.0] * 10, _last_value, lookback=4, warmup_ticks=4)
    assert confidence == pytest.approx(1.0)


def test_confidence_falls_when_recent_errors_grow() -> None:
    """A series the model keeps missing scores lower than one it tracks."""
    steady = recent_error_confidence([2.0] * 12, _last_value, lookback=6, warmup_ticks=4)
    erratic = recent_error_confidence(
        [2.0, 9.0, 1.0, 8.0, 0.5, 7.0, 1.5, 9.5], _last_value, lookback=6, warmup_ticks=4
    )
    assert erratic < steady


def test_confidence_is_capped_by_the_warmup_ramp() -> None:
    """Short history can't score high even when the few predictions were perfect."""
    confidence = recent_error_confidence([3.0, 3.0, 3.0], _last_value, lookback=8, warmup_ticks=12)
    assert confidence == pytest.approx(3 / 12)


def test_confidence_stays_within_bounds() -> None:
    """Whatever the error, the score respects the ForecastOutput contract's [0, 1]."""
    awful = recent_error_confidence(
        [0.1, 50.0, 0.1, 50.0, 0.1, 50.0], _last_value, lookback=5, warmup_ticks=2
    )
    assert 0.0 <= awful <= 1.0
    assert awful == 0.0
