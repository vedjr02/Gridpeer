"""Hold-out backtest: naive baseline vs LightGBM, per household.

This produces the forecasting number for the writeup — "our demand forecaster cuts
MAE from X to Y kWh per half-hour against the naive baseline" — rather than an
internal sanity print. Run it before wiring a new model into the live pipeline.

Evaluation is expanding-window rolling origin over the last `holdout_days`: every
test tick is predicted from that household's history *strictly preceding it*, and
the GBM refits on its own schedule as history accrues. That is exactly how the
orchestrator drives the model, so the reported error is the error the pipeline
will actually see — and no model ever sees a tick before predicting it.

    python -m forecasting.backtest

Caveat for the writeup: against `data/synthetic.py` the GBM is scored on a smooth
generated shape plus bounded noise, which it learns very well. Treat the margin
here as an upper bound and re-run this against real CER data before quoting a
number — the comparison is honest, the difficulty is not yet realistic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from data.synthetic import TICK_MINUTES, default_profiles, generate_household_series, ticks_per_day
from forecasting.base import Forecaster
from forecasting.baseline import NaiveForecaster
from forecasting.gbm import GBMForecaster
from forecasting.metrics import mae, mape, normalised_mae

# A Monday, so weekday/weekend features line up with the synthetic generator's
# `day % 7 >= 5` weekend rule.
BACKTEST_EPOCH = datetime(2026, 1, 5)


@dataclass(frozen=True)
class SeriesErrors:
    """Error summary for one predicted quantity over the holdout window."""

    mae_kwh: float
    normalised_mae: float
    mape_pct: float | None  # None where the actuals sit at ~0 and MAPE is undefined


@dataclass(frozen=True)
class BacktestResult:
    """One model's holdout performance for one household (or pooled across all)."""

    model_name: str
    household_id: str
    num_test_ticks: int
    demand: SeriesErrors
    solar: SeriesErrors


def _errors(actual_kwh: Sequence[float], predicted_kwh: Sequence[float]) -> SeriesErrors:
    """Summarise one quantity's error, tolerating series that legitimately hit zero."""
    try:
        mape_pct: float | None = mape(actual_kwh, predicted_kwh)
    except ValueError:
        mape_pct = None
    return SeriesErrors(
        mae_kwh=mae(actual_kwh, predicted_kwh),
        normalised_mae=normalised_mae(actual_kwh, predicted_kwh),
        mape_pct=mape_pct,
    )


def rolling_predictions(
    forecaster: Forecaster,
    household_id: str,
    demand_kwh: Sequence[float],
    solar_kwh: Sequence[float],
    first_test_tick: int,
    epoch: datetime = BACKTEST_EPOCH,
    tick_minutes: int = TICK_MINUTES,
) -> tuple[list[float], list[float]]:
    """Walk the holdout forward, predicting each tick from history before it only."""
    demand_predictions: list[float] = []
    solar_predictions: list[float] = []
    for tick in range(first_test_tick, len(demand_kwh)):
        output = forecaster.forecast(
            household_id=household_id,
            tick=tick,
            timestamp=epoch + timedelta(minutes=tick_minutes * tick),
            demand_history_kwh=demand_kwh[:tick],
            solar_history_kwh=solar_kwh[:tick],
        )
        demand_predictions.append(output.predicted_demand_kwh)
        solar_predictions.append(output.predicted_solar_generation_kwh)
    return demand_predictions, solar_predictions


def backtest_models(
    models: dict[str, Callable[[], Forecaster]],
    num_households: int = 4,
    days: int = 28,
    holdout_days: int = 7,
    seed: int = 0,
) -> list[BacktestResult]:
    """Backtest each model on a synthetic cohort; returns per-household + pooled results.

    `models` maps a display name to a *factory*, because a fitted forecaster carries
    state — each household must get its own instance or the comparison is meaningless.
    """
    if holdout_days >= days:
        raise ValueError("holdout_days must be smaller than days")
    per_day = ticks_per_day(TICK_MINUTES)
    first_test_tick = (days - holdout_days) * per_day

    profiles = default_profiles(num_households)
    series = {
        profile.household_id: generate_household_series(profile, days=days, seed=seed)
        for profile in profiles
    }

    results: list[BacktestResult] = []
    for model_name, make_forecaster in models.items():
        pooled_demand_actual: list[float] = []
        pooled_demand_pred: list[float] = []
        pooled_solar_actual: list[float] = []
        pooled_solar_pred: list[float] = []

        for profile in profiles:
            demand_kwh, solar_kwh = series[profile.household_id]
            demand_pred, solar_pred = rolling_predictions(
                make_forecaster(),
                profile.household_id,
                demand_kwh,
                solar_kwh,
                first_test_tick,
            )
            demand_actual = list(demand_kwh[first_test_tick:])
            solar_actual = list(solar_kwh[first_test_tick:])
            results.append(
                BacktestResult(
                    model_name=model_name,
                    household_id=profile.household_id,
                    num_test_ticks=len(demand_actual),
                    demand=_errors(demand_actual, demand_pred),
                    solar=_errors(solar_actual, solar_pred),
                )
            )
            pooled_demand_actual += demand_actual
            pooled_demand_pred += demand_pred
            pooled_solar_actual += solar_actual
            pooled_solar_pred += solar_pred

        results.append(
            BacktestResult(
                model_name=model_name,
                household_id="ALL",
                num_test_ticks=len(pooled_demand_actual),
                demand=_errors(pooled_demand_actual, pooled_demand_pred),
                solar=_errors(pooled_solar_actual, pooled_solar_pred),
            )
        )
    return results


def _cell(value: float | None, suffix: str = "") -> str:
    """Render a metric, or a dash where it is undefined."""
    return "     —" if value is None else f"{value:6.3f}{suffix}"


def format_table(results: Sequence[BacktestResult]) -> str:
    """Render the model comparison as a fixed-width table for the writeup."""
    header = (
        f"{'model':<10} {'household':<10} {'ticks':>6} "
        f"{'dem MAE':>9} {'dem nMAE':>9} {'dem MAPE':>9} {'sol MAE':>9} {'sol nMAE':>9}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(
            f"{result.model_name:<10} {result.household_id:<10} {result.num_test_ticks:>6} "
            f"{_cell(result.demand.mae_kwh):>9} {_cell(result.demand.normalised_mae):>9} "
            f"{_cell(result.demand.mape_pct):>9} "
            f"{_cell(result.solar.mae_kwh):>9} {_cell(result.solar.normalised_mae):>9}"
        )
    return "\n".join(lines)


def _improvement_line(results: Sequence[BacktestResult]) -> str:
    """The one-sentence headline: how much the GBM beats the naive baseline by."""
    pooled = {r.model_name: r for r in results if r.household_id == "ALL"}
    if not {"naive", "lightgbm"} <= pooled.keys():
        return ""
    naive_mae = pooled["naive"].demand.mae_kwh
    gbm_mae = pooled["lightgbm"].demand.mae_kwh
    delta_pct = 100.0 * (naive_mae - gbm_mae) / naive_mae
    direction = "better" if delta_pct >= 0 else "WORSE"
    return (
        f"\nDemand MAE pooled over all households: naive {naive_mae:.4f} kWh -> "
        f"lightgbm {gbm_mae:.4f} kWh  ({abs(delta_pct):.1f}% {direction})."
    )


def main() -> None:
    """Run the naive-vs-LightGBM backtest and print the comparison table."""
    models: dict[str, Callable[[], Forecaster]] = {
        "naive": NaiveForecaster,
        "lightgbm": GBMForecaster,
    }
    results = backtest_models(models)
    print("Holdout backtest — expanding-window rolling origin, last 7 of 28 days")
    print("MAE/nMAE in kWh per half-hour tick; nMAE = MAE / mean actual level.\n")
    print(format_table(results))
    print(_improvement_line(results))


if __name__ == "__main__":
    main()
