"""Naive baseline forecaster: rolling-average demand + solar prediction.

Build-order step 1 (see forecasting/AGENTS.md): the boring sanity-check model
that stays in the codebase as a fallback once smarter models (LightGBM/Prophet)
arrive behind the same ForecastOutput contract. It predicts the next tick as the
mean of the most recent observations, falling back to a flat value when no
history exists yet — enough to make the end-to-end pipeline real in week 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from functools import partial

from forecasting.metrics import recent_error_confidence
from shared.schemas import ForecastOutput


class NaiveForecaster:
    """Rolling-average forecaster for one household's demand and solar output.

    Predicts each quantity as the mean of the last ``window`` observations.
    Not accurate — it is the reference point every real model must beat.
    """

    def __init__(
        self,
        window: int = 4,
        flat_demand_kwh: float = 0.5,
        confidence_lookback: int = 8,
    ) -> None:
        """window: how many recent observations to average (half-hour ticks).
        flat_demand_kwh: fallback demand when no history is available yet.
        confidence_lookback: how many recent ticks the confidence score is scored over.
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        if flat_demand_kwh < 0:
            raise ValueError("flat_demand_kwh must be >= 0")
        if confidence_lookback < 1:
            raise ValueError("confidence_lookback must be >= 1")
        self.window = window
        self.flat_demand_kwh = flat_demand_kwh
        self.confidence_lookback = confidence_lookback

    def _rolling_mean(self, history_kwh: Sequence[float], fallback_kwh: float) -> float:
        """Mean of the last ``window`` values in kWh; fallback if history empty."""
        if not history_kwh:
            return fallback_kwh
        recent = history_kwh[-self.window :]
        return sum(recent) / len(recent)

    def _confidence(self, history_kwh: Sequence[float], fallback_kwh: float) -> float:
        """Confidence in [0, 1] from this model's own normalised error on recent history.

        Replays the rolling average over the last `confidence_lookback` ticks and
        scores 1 minus its mean normalised error, capped by a warm-up ramp. Agents
        can read this to bid more conservatively when the forecast is unreliable.
        """
        predict = partial(self._rolling_mean, fallback_kwh=fallback_kwh)
        return recent_error_confidence(
            history_kwh,
            predict=predict,
            lookback=self.confidence_lookback,
            warmup_ticks=self.window,
        )

    def forecast(
        self,
        household_id: str,
        tick: int,
        timestamp: datetime,
        demand_history_kwh: Sequence[float],
        solar_history_kwh: Sequence[float],
    ) -> ForecastOutput:
        """Predict one household's next-tick demand and solar generation (kWh).

        History sequences are past observations in chronological order. Returns a
        ForecastOutput satisfying the shared contract — the only thing agents/ and
        dashboard/ consume from this module.
        """
        predicted_demand = self._rolling_mean(demand_history_kwh, self.flat_demand_kwh)
        predicted_solar = self._rolling_mean(solar_history_kwh, 0.0)
        confidence = min(
            self._confidence(demand_history_kwh, self.flat_demand_kwh),
            self._confidence(solar_history_kwh, 0.0),
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
