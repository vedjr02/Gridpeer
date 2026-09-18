"""LightGBM demand forecaster, behind the same ForecastOutput contract as the baseline.

Build-order step 2 (see `forecasting/AGENTS.md`): gradient boosting on lag and
calendar features, which at CER data volumes (half-hourly, per household) beats a
neural net and trains in seconds. Solar is deliberately *not* boosted — generation
is near-deterministic given time of day and panel capacity, so it uses a
tick-of-day climatology (step 3: "don't over-invest here relative to demand").

Drop-in for `NaiveForecaster`: same `forecast()` signature, so the orchestrator
swaps one constructor. The model refits itself from the history it is handed and
falls back to the naive rolling average until there is enough of it, which keeps
the pipeline running from tick 0 instead of erroring on a cold start.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from forecasting.metrics import recent_error_confidence
from shared.schemas import ForecastOutput

TICKS_PER_DAY = 48  # half-hourly, matching CER resolution
LAG_TICKS = (1, 2, 3, 4, TICKS_PER_DAY, 2 * TICKS_PER_DAY)
ROLLING_WINDOWS = (4, 12, TICKS_PER_DAY)
MAX_LAG = max(max(LAG_TICKS), max(ROLLING_WINDOWS))

FEATURE_NAMES = (
    [f"lag_{lag}" for lag in LAG_TICKS]
    + [f"roll_mean_{window}" for window in ROLLING_WINDOWS]
    + ["tick_of_day", "tick_of_day_sin", "tick_of_day_cos", "day_of_week", "is_weekend"]
)

COLD_START_DEMAND_KWH = 0.5  # flat demand before any history exists, matching the baseline
COLD_START_WINDOW = 4


def _rolling_mean(history_kwh: Sequence[float], fallback_kwh: float) -> float:
    """Mean of the last few observations (kWh) — the cold-start path before the model fits."""
    if not history_kwh:
        return fallback_kwh
    recent = history_kwh[-COLD_START_WINDOW:]
    return sum(recent) / len(recent)


# LightGBM's native training API, not the sklearn wrapper — the wrapper would drag
# scikit-learn onto every teammate's machine for no gain here.
NUM_BOOST_ROUNDS = 300
_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression_l1",  # L1 tracks MAE, which is what the backtest reports
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 10,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "feature_fraction": 0.9,
    "verbosity": -1,
    "force_row_wise": True,
    "deterministic": True,
}


class GBMForecaster:
    """Gradient-boosted demand forecaster with a climatology solar model."""

    def __init__(
        self,
        tick_minutes: int = 30,
        min_train_ticks: int = 4 * TICKS_PER_DAY,
        refit_every: int = TICKS_PER_DAY,
        solar_lookback_days: int = 7,
        confidence_lookback: int = 8,
        random_state: int = 0,
    ) -> None:
        """min_train_ticks: history needed before the first fit; until then, naive fallback.
        refit_every: refit after this many further ticks of history accrue.
        """
        if min_train_ticks <= MAX_LAG:
            raise ValueError(f"min_train_ticks must exceed the longest lag ({MAX_LAG})")
        if refit_every < 1 or solar_lookback_days < 1 or confidence_lookback < 1:
            raise ValueError("refit_every, solar_lookback_days, confidence_lookback must be >= 1")
        self.tick_minutes = tick_minutes
        self.min_train_ticks = min_train_ticks
        self.refit_every = refit_every
        self.solar_lookback_days = solar_lookback_days
        self.confidence_lookback = confidence_lookback
        self.random_state = random_state

        self._model: Any | None = None
        self._trained_at_ticks = 0

    # -- features -----------------------------------------------------------
    def _row(
        self, history_kwh: Sequence[float], index: int, base_timestamp: datetime
    ) -> list[float]:
        """Feature row for the tick at `index`, using only observations before it."""
        row = [float(history_kwh[index - lag]) for lag in LAG_TICKS]
        for window in ROLLING_WINDOWS:
            recent = history_kwh[index - window : index]
            row.append(sum(recent) / len(recent))
        moment = base_timestamp + timedelta(minutes=self.tick_minutes * index)
        tick_of_day = (moment.hour * 60 + moment.minute) // self.tick_minutes
        angle = 2.0 * math.pi * tick_of_day / TICKS_PER_DAY
        row += [
            float(tick_of_day),
            math.sin(angle),
            math.cos(angle),
            float(moment.weekday()),
            float(moment.weekday() >= 5),
        ]
        return row

    def _base_timestamp(self, tick: int, timestamp: datetime) -> datetime:
        """Wall-clock time of tick 0, so history indices can be given calendar features."""
        return timestamp - timedelta(minutes=self.tick_minutes * tick)

    # -- training -----------------------------------------------------------
    def fit(self, demand_history_kwh: Sequence[float], base_timestamp: datetime) -> None:
        """Train the demand model on every tick of history that has full lag coverage."""
        if len(demand_history_kwh) <= MAX_LAG:
            raise ValueError(f"need more than {MAX_LAG} ticks of history to fit")
        import lightgbm as lgb  # imported lazily: the naive path must work without it

        features = np.array(
            [
                self._row(demand_history_kwh, index, base_timestamp)
                for index in range(MAX_LAG, len(demand_history_kwh))
            ],
            dtype=float,
        )
        targets = np.array(demand_history_kwh[MAX_LAG:], dtype=float)
        dataset = lgb.Dataset(features, label=targets, feature_name=list(FEATURE_NAMES))
        self._model = lgb.train(
            {**_LGBM_PARAMS, "seed": self.random_state},
            dataset,
            num_boost_round=NUM_BOOST_ROUNDS,
        )
        self._trained_at_ticks = len(demand_history_kwh)

    def _maybe_fit(self, demand_history_kwh: Sequence[float], base_timestamp: datetime) -> None:
        """Fit on the first sufficient history, then refit every `refit_every` ticks."""
        available = len(demand_history_kwh)
        if available < self.min_train_ticks:
            return
        if self._model is not None and available - self._trained_at_ticks < self.refit_every:
            return
        self.fit(demand_history_kwh, base_timestamp)

    # -- prediction ---------------------------------------------------------
    def _predict_demand(self, history_kwh: Sequence[float], base_timestamp: datetime) -> float:
        """Predicted demand (kWh) for the tick immediately after `history_kwh`."""
        if self._model is None or len(history_kwh) < MAX_LAG:
            return _rolling_mean(history_kwh, COLD_START_DEMAND_KWH)
        row = np.array([self._row(history_kwh, len(history_kwh), base_timestamp)], dtype=float)
        return max(0.0, float(self._model.predict(row)[0]))

    def _predict_solar(self, history_kwh: Sequence[float], tick: int) -> float:
        """Predicted solar (kWh) as the mean of recent same-time-of-day observations."""
        same_time = [
            history_kwh[index]
            for index in range(tick - TICKS_PER_DAY, -1, -TICKS_PER_DAY)
            if 0 <= index < len(history_kwh)
        ][: self.solar_lookback_days]
        if same_time:
            return max(0.0, sum(same_time) / len(same_time))
        recent = history_kwh[-4:]
        return max(0.0, sum(recent) / len(recent)) if recent else 0.0

    def forecast(
        self,
        household_id: str,
        tick: int,
        timestamp: datetime,
        demand_history_kwh: Sequence[float],
        solar_history_kwh: Sequence[float],
    ) -> ForecastOutput:
        """Predict one household's next-tick demand and solar generation (kWh)."""
        base_timestamp = self._base_timestamp(tick, timestamp)
        self._maybe_fit(demand_history_kwh, base_timestamp)

        predicted_demand = self._predict_demand(demand_history_kwh, base_timestamp)
        predicted_solar = self._predict_solar(solar_history_kwh, tick)
        confidence = min(
            recent_error_confidence(
                demand_history_kwh,
                predict=lambda partial_history: self._predict_demand(
                    partial_history, base_timestamp
                ),
                lookback=self.confidence_lookback,
                warmup_ticks=MAX_LAG,
            ),
            recent_error_confidence(
                solar_history_kwh,
                predict=lambda partial_history: self._predict_solar(
                    partial_history, len(partial_history)
                ),
                lookback=self.confidence_lookback,
                warmup_ticks=MAX_LAG,
            ),
        )
        return ForecastOutput(
            household_id=household_id,
            tick=tick,
            timestamp=timestamp,
            predicted_demand_kwh=predicted_demand,
            predicted_solar_generation_kwh=predicted_solar,
            predicted_net_position_kwh=predicted_solar - predicted_demand,
            confidence=confidence,
        )
