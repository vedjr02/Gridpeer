"""Forecast error metrics for backtesting (MAE, MAPE).

Reported per household when holding out the last N days of history
(see forecasting/CLAUDE.md, "Backtesting"). These become genuinely
reportable numbers in the final writeup, not just an internal check.
"""

from __future__ import annotations

from collections.abc import Sequence


def _validate(actual: Sequence[float], predicted: Sequence[float]) -> None:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must be the same length")
    if not actual:
        raise ValueError("cannot compute error over an empty series")


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error, in the same unit as the inputs (kWh)."""
    _validate(actual, predicted)
    return sum(abs(a - p) for a, p in zip(actual, predicted, strict=True)) / len(actual)


def mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute percentage error, as a fraction (0.1 == 10%).

    Points where actual == 0 are excluded: percentage error is undefined
    there. Returns 0.0 when every actual is zero (nothing measurable).
    """
    _validate(actual, predicted)
    errors = [abs(a - p) / abs(a) for a, p in zip(actual, predicted, strict=True) if a != 0]
    if not errors:
        return 0.0
    return sum(errors) / len(errors)
