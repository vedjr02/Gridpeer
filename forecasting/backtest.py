"""Walk-forward backtest for forecasters, reporting MAE/MAPE per household.

Holds out no data explicitly — instead it walks the series forward: at each
tick it forecasts using only the history available up to that point, then
compares against the actual value. This mirrors how the forecaster is used in
the live pipeline (it never sees the future) and yields honest error numbers.

Run a demo report:  python -m forecasting.backtest
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from forecasting.baseline import NaiveForecaster
from forecasting.metrics import mae, mape

# Half-hourly ticks, matching the CER smart-meter resolution.
TICK_MINUTES = 30
_EPOCH = datetime(2026, 1, 1, 0, 0)


def _timestamp(tick: int) -> datetime:
    """Wall-clock time for a tick, at half-hour resolution."""
    return _EPOCH + timedelta(minutes=TICK_MINUTES * tick)


@dataclass(frozen=True)
class BacktestResult:
    """Per-household forecast error over a walk-forward backtest.

    Errors are in kWh (MAE) and as fractions (MAPE, 0.1 == 10%). ``n_points``
    is how many ticks were scored (the warm-up history is not scored).
    """

    household_id: str
    n_points: int
    demand_mae: float
    demand_mape: float
    solar_mae: float
    solar_mape: float


def backtest_household(
    forecaster: NaiveForecaster,
    household_id: str,
    demand_kwh: list[float],
    solar_kwh: list[float],
    min_history: int = 1,
) -> BacktestResult:
    """Walk ``forecaster`` forward over one household's series and score it.

    For each tick t >= ``min_history``, forecast using only observations before
    t, then compare against the actual demand and solar at t. Demand and solar
    series must be the same length and at least ``min_history`` + 1 long.
    """
    if len(demand_kwh) != len(solar_kwh):
        raise ValueError("demand and solar series must be the same length")
    if min_history < 1:
        raise ValueError("min_history must be >= 1")
    n = len(demand_kwh)
    if n <= min_history:
        raise ValueError("series too short to score any tick after warm-up")

    pred_demand: list[float] = []
    pred_solar: list[float] = []
    actual_demand: list[float] = []
    actual_solar: list[float] = []

    for t in range(min_history, n):
        forecast = forecaster.forecast(
            household_id=household_id,
            tick=t,
            timestamp=_timestamp(t),
            demand_history_kwh=demand_kwh[:t],
            solar_history_kwh=solar_kwh[:t],
        )
        pred_demand.append(forecast.predicted_demand_kwh)
        pred_solar.append(forecast.predicted_solar_generation_kwh)
        actual_demand.append(demand_kwh[t])
        actual_solar.append(solar_kwh[t])

    return BacktestResult(
        household_id=household_id,
        n_points=len(actual_demand),
        demand_mae=mae(actual_demand, pred_demand),
        demand_mape=mape(actual_demand, pred_demand),
        solar_mae=mae(actual_solar, pred_solar),
        solar_mape=mape(actual_solar, pred_solar),
    )


def _demo_series(days: int = 3, seed: int = 0) -> tuple[list[float], list[float]]:
    """Self-contained placeholder series (no data/ dependency) for the demo.

    Demand: morning + evening peaks over a flat base. Solar: a daylight bell
    curve, zero overnight. Shapes only — real CER/PVGIS data replaces this.
    """
    ticks_per_day = 24 * 60 // TICK_MINUTES  # 48
    demand: list[float] = []
    solar: list[float] = []
    for tick in range(days * ticks_per_day):
        hour = (tick % ticks_per_day) * TICK_MINUTES / 60.0
        # deterministic pseudo-noise, no imports needed
        jitter = ((tick * 2654435761 + seed) % 1000) / 1000.0 * 0.2
        morning = math.exp(-((hour - 8) ** 2) / 2.0)
        evening = math.exp(-((hour - 19) ** 2) / 2.0)
        demand.append(0.3 + 1.2 * (morning + evening) + jitter)
        midday = math.exp(-((hour - 13) ** 2) / 8.0)
        solar.append(max(0.0, 2.5 * midday) if 6 <= hour <= 20 else 0.0)
    return demand, solar


def main() -> None:
    """Print a demo backtest report for the naive baseline forecaster."""
    demand, solar = _demo_series()
    result = backtest_household(NaiveForecaster(window=4), "demo_household", demand, solar)
    print(f"Backtest — {result.household_id} (n={result.n_points} ticks scored)")
    print(f"  demand: MAE={result.demand_mae:.3f} kWh  MAPE={result.demand_mape * 100:.1f}%")
    print(f"  solar:  MAE={result.solar_mae:.3f} kWh  MAPE={result.solar_mape * 100:.1f}%")


if __name__ == "__main__":
    main()
