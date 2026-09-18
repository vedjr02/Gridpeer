"""Forecast detail tab: the model-diagnostic charts, off the main view.

These four series say how the forecaster is behaving, not whether households
saved money, so by the rule in dashboard/CLAUDE.md they do not belong beside
the headline claims. They stay because they are how the forecasting module is
debugged — just one tab away from the narrative.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.views import theme
from shared.schemas import ForecastOutput


def render(forecasts: list[ForecastOutput], palette: theme.HouseholdPalette) -> None:
    """Draw the four forecast series and the raw stream for the selected run."""
    st.subheader("Forecasting stage")
    st.caption(
        "The ForecastOutput stream this run was driven by, as persisted — each "
        "household forecast from its own history up to that tick only. Diagnostics "
        "for forecasting/; these do not support the savings or comparison claims."
    )

    if not forecasts:
        st.info("This run has no persisted forecasts.")
        return

    frame = _to_frame(forecasts)

    _series_chart(
        frame,
        palette,
        "predicted_demand_kwh",
        f"Predicted demand ({theme.KWH})",
        f"Demand ({theme.KWH})",
    )
    _series_chart(
        frame,
        palette,
        "predicted_solar_generation_kwh",
        f"Predicted solar generation ({theme.KWH})",
        f"Generation ({theme.KWH})",
    )
    _series_chart(
        frame,
        palette,
        "predicted_net_position_kwh",
        f"Predicted net position ({theme.KWH}) — positive = surplus to sell",
        f"Net position ({theme.KWH})",
        zero_rule=True,
    )
    _series_chart(
        frame,
        palette,
        "confidence",
        "Forecast confidence (0–1)",
        "Confidence (0–1)",
    )

    with st.expander("Raw ForecastOutput stream"):
        st.dataframe(frame, hide_index=True, width="stretch")


def _to_frame(stream: list[ForecastOutput]) -> pd.DataFrame:
    """Flatten the ForecastOutput stream into a tidy frame for charting."""
    return pd.DataFrame(
        {
            "tick": f.tick,
            "household_id": f.household_id,
            "predicted_demand_kwh": f.predicted_demand_kwh,
            "predicted_solar_generation_kwh": f.predicted_solar_generation_kwh,
            "predicted_net_position_kwh": f.predicted_net_position_kwh,
            "confidence": f.confidence,
        }
        for f in stream
    )


def _series_chart(
    frame: pd.DataFrame,
    palette: theme.HouseholdPalette,
    column: str,
    heading: str,
    y_title: str,
    zero_rule: bool = False,
) -> None:
    """Render one per-household line chart, colours matching every other view."""
    st.markdown(f"**{heading}**")
    chart = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2)
        .encode(
            x=theme.tick_axis(),
            y=alt.Y(f"{column}:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=palette.colour(),
            tooltip=[
                alt.Tooltip("household_id:N", title="Household"),
                alt.Tooltip("tick:Q", title="Tick"),
                alt.Tooltip(f"{column}:Q", title=y_title, format=".3f"),
            ],
        )
    )
    if zero_rule:
        baseline = (
            alt.Chart(pd.DataFrame({"zero": [0.0]}))
            .mark_rule(strokeWidth=1, color=theme.REFERENCE_RULE_HUE)
            .encode(y="zero:Q")
        )
        chart = chart + baseline
    st.altair_chart(theme.style(chart, height=240), width="stretch")
