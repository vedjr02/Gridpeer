"""Tests for forecast error metrics."""

import pytest

from forecasting.metrics import mae, mape


def test_mae_basic() -> None:
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert mae([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.5)


def test_mape_basic() -> None:
    assert mape([2.0, 4.0], [1.0, 2.0]) == pytest.approx(0.5)


def test_mape_excludes_zero_actuals() -> None:
    # First point (actual 0) excluded; only the second counts.
    assert mape([0.0, 4.0], [1.0, 2.0]) == pytest.approx(0.5)


def test_mape_all_zero_actuals_returns_zero() -> None:
    assert mape([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        mae([1.0], [1.0, 2.0])


def test_empty_series_raises() -> None:
    with pytest.raises(ValueError):
        mape([], [])
