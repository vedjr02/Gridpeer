"""Forecast error metrics and the confidence score every forecaster reports.

Kept in one place because the models, their tests and the backtest all need the
same definitions — a MAE that means something different in `backtest.py` than in
`baseline.py` would quietly invalidate the comparison table.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

# Denominator floor (kWh) for normalised errors: stops a near-zero actual — an
# overnight solar reading, say — turning a tiny absolute miss into a huge ratio.
ERROR_SCALE_FLOOR_KWH = 0.05


def mae(actual_kwh: Sequence[float], predicted_kwh: Sequence[float]) -> float:
    """Mean absolute error in kWh."""
    if len(actual_kwh) != len(predicted_kwh):
        raise ValueError("actual and predicted must be the same length")
    if not actual_kwh:
        raise ValueError("cannot compute MAE over an empty series")
    return sum(abs(a - p) for a, p in zip(actual_kwh, predicted_kwh, strict=True)) / len(actual_kwh)


def mape(actual_kwh: Sequence[float], predicted_kwh: Sequence[float]) -> float:
    """Mean absolute percentage error (%), skipping actuals at or below the scale floor.

    Undefined where the actual is ~0, which is most of a solar series overnight —
    use `normalised_mae` for quantities that legitimately sit at zero.
    """
    if len(actual_kwh) != len(predicted_kwh):
        raise ValueError("actual and predicted must be the same length")
    pairs = [
        (a, p)
        for a, p in zip(actual_kwh, predicted_kwh, strict=True)
        if abs(a) > ERROR_SCALE_FLOOR_KWH
    ]
    if not pairs:
        raise ValueError("no actuals above the scale floor — MAPE is undefined here")
    return 100.0 * sum(abs(a - p) / abs(a) for a, p in pairs) / len(pairs)


def normalised_mae(actual_kwh: Sequence[float], predicted_kwh: Sequence[float]) -> float:
    """MAE as a fraction of the series' own mean level — safe when actuals hit zero."""
    scale = max(sum(abs(a) for a in actual_kwh) / len(actual_kwh), ERROR_SCALE_FLOOR_KWH)
    return mae(actual_kwh, predicted_kwh) / scale


def recent_error_confidence(
    history_kwh: Sequence[float],
    predict: Callable[[Sequence[float]], float],
    lookback: int,
    warmup_ticks: int,
) -> float:
    """Confidence in [0, 1] from the model's own normalised error on recent history.

    Replays `predict` against the last `lookback` ticks — each one predicted from
    only the data that preceded it — and scores confidence as 1 minus the mean
    normalised error. Capped by a warm-up ramp so a model with almost no history
    can never report high confidence on the strength of one lucky tick.
    """
    if lookback < 1 or warmup_ticks < 1:
        raise ValueError("lookback and warmup_ticks must be >= 1")
    ramp = min(len(history_kwh), warmup_ticks) / warmup_ticks
    if len(history_kwh) < 2:
        return 0.0

    start = max(1, len(history_kwh) - lookback)
    actuals = list(history_kwh[start:])
    predictions = [predict(history_kwh[:index]) for index in range(start, len(history_kwh))]
    if not actuals:
        return 0.0
    accuracy = max(0.0, 1.0 - normalised_mae(actuals, predictions))
    return min(ramp, accuracy)
