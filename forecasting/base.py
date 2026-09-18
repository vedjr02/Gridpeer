"""The contract every forecaster in this module satisfies.

`agents/` and `dashboard/` only ever see `ForecastOutput`; this Protocol is what
lets the orchestrator swap NaiveForecaster for GBMForecaster (or anything later)
without touching a line of the pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from shared.schemas import ForecastOutput


@runtime_checkable
class Forecaster(Protocol):
    """Predicts one household's next-tick demand and solar generation."""

    def forecast(
        self,
        household_id: str,
        tick: int,
        timestamp: datetime,
        demand_history_kwh: Sequence[float],
        solar_history_kwh: Sequence[float],
    ) -> ForecastOutput:
        """Return a contract-valid ForecastOutput for `tick` from history before it.

        History sequences are chronological observations in kWh strictly preceding
        `tick`; an implementation must never look at or beyond the tick it predicts.
        """
        ...
