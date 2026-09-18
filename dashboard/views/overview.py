"""Main view: the two claims the demo has to defend, in priority order.

Claim 1 — households saved X% versus just exporting to the grid.
Claim 2 — RL agents beat the rule-based baseline by Y% (the Strategy comparison
tab carries this one; here the headline row states the single run's number).

Everything on this view supports one of those. Charts that do not earn their
place here live in the Forecast detail tab instead.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.views import theme, transforms
from dashboard.views.data import RunData, tariff_for


def render(run: RunData) -> None:
    """Draw the headline metric row and the three outcome charts for one run."""
    _headline_metrics(run)
    st.divider()
    _savings_chart(run)
    _clearing_price_chart(run)
    _trade_volume_chart(run)
    _outcomes_table(run)


def _headline_metrics(run: RunData) -> None:
    """Four numbers the demo is judged on, stated before any chart."""
    summary = run.summary
    st.subheader(f"Run outcome — {summary.strategy_name}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        f"Total savings ({theme.EUR})",
        f"{summary.total_savings_eur:,.2f}",
        help="Grid-only bill minus the P2P bill, summed across households.",
    )
    col2.metric(
        "Avg savings (%)",
        f"{summary.avg_savings_pct:.1f}%",
        help="Mean per-household saving versus importing and exporting at grid tariffs.",
    )
    col3.metric(
        "CO2 avoided (kg)",
        f"{summary.co2_avoided_kg:,.1f}",
        help="Attributed to energy met locally instead of drawn from the grid.",
    )
    col4.metric(
        "Peak-load reduction (%)",
        f"{summary.peak_load_reduction_pct:.1f}%",
        help="Drop in peak grid draw versus the same households trading with the grid only.",
    )
    st.caption(
        f"{summary.num_households} households · {summary.num_ticks} half-hourly ticks · "
        f"run `{summary.run_id}`"
    )


def _savings_chart(run: RunData) -> None:
    """Priority chart: what each household paid P2P versus the grid-only baseline.

    Read straight from HouseholdOutcome so this chart and the headline metric row
    can never quote different numbers — outcomes.py owns that arithmetic.
    """
    st.subheader(f"Household cost: peer-to-peer vs grid-only baseline ({theme.EUR})")
    st.caption(
        "Cost is positive when money leaves the household, so a net producer sits "
        "below zero — that bar is revenue. A shorter P2P bar than grid-only bar is "
        "the saving; the exact figure is on each pair and in the table below."
    )

    frame = transforms.savings_by_household(run.outcomes)
    if frame.empty:
        st.info("This run has no per-household outcomes stored yet.")
        return

    base = alt.Chart(frame)
    bars = base.mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("basis:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("cost_eur:Q", title=f"Run-long cost ({theme.EUR})"),
        color=theme.basis_colour(),
        column=alt.Column("household_id:N", title="Household", spacing=14),
        tooltip=[
            alt.Tooltip("household_id:N", title="Household"),
            alt.Tooltip("basis:N", title="Basis"),
            alt.Tooltip("cost_eur:Q", title=f"Cost ({theme.EUR})", format=",.2f"),
            alt.Tooltip("savings_eur:Q", title=f"Saved ({theme.EUR})", format=",.2f"),
            alt.Tooltip("savings_pct:Q", title="Saved (%)", format=",.1f"),
        ],
    )
    st.altair_chart(theme.style_faceted(bars, height=260), width="stretch")

    saved = (
        frame[["household_id", "savings_eur", "savings_pct"]]
        .drop_duplicates("household_id")
        .sort_values("household_id")
    )
    st.caption(
        "Saved per household:  "
        + "   ·   ".join(
            f"**{row.household_id}** {row.savings_eur:+,.2f} {theme.EUR} "
            f"({row.savings_pct:+.1f}%)"
            for row in saved.itertuples()
        )
    )


def _clearing_price_chart(run: RunData) -> None:
    """Clearing price per tick, read against the grid tariff band it must sit in."""
    st.subheader(f"Clearing price over time ({theme.EUR_PER_KWH})")
    st.caption(
        "A price between the export and import tariff is the whole mechanism: the "
        "seller beats what the grid would pay them, the buyer beats what the grid "
        "would charge them. Ticks that cleared no trades are gaps, not zeros."
    )

    frame = transforms.clearing_price_per_tick(run.market_states)
    if frame.empty:
        st.info("No tick in this run cleared a trade, so there is no price series yet.")
        return

    import_tariff, export_tariff = _tariff_band(run)
    price_line = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2, color=theme.P2P_HUE)
        .encode(
            x=theme.tick_axis(),
            y=alt.Y(
                "clearing_price_eur_per_kwh:Q",
                title=f"Clearing price ({theme.EUR_PER_KWH})",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("tick:Q", title="Tick"),
                alt.Tooltip(
                    "clearing_price_eur_per_kwh:Q",
                    title=f"Price ({theme.EUR_PER_KWH})",
                    format=".3f",
                ),
            ],
        )
    )
    bands = pd.DataFrame(
        [
            {"level_eur_per_kwh": import_tariff, "label": "Grid import tariff"},
            {"level_eur_per_kwh": export_tariff, "label": "Grid export tariff"},
        ]
    )
    rules = (
        alt.Chart(bands)
        .mark_rule(strokeDash=[4, 4], strokeWidth=1, color=theme.REFERENCE_RULE_HUE)
        .encode(y="level_eur_per_kwh:Q")
    )
    labels = (
        alt.Chart(bands)
        .mark_text(align="left", dx=6, dy=-6, fontSize=11, color=theme.REFERENCE_RULE_HUE)
        .encode(y="level_eur_per_kwh:Q", text="label:N")
    )
    st.altair_chart(theme.style(price_line + rules + labels), width="stretch")


def _trade_volume_chart(run: RunData) -> None:
    """How much energy actually changed hands each tick."""
    st.subheader(f"Trade volume per tick ({theme.KWH})")
    st.caption("Energy cleared peer-to-peer each half-hour — the activity behind the savings.")

    frame = transforms.volume_per_tick(run.trades, run.summary.num_ticks)
    chart = (
        alt.Chart(frame)
        .mark_bar(color=theme.P2P_HUE, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=theme.tick_axis(),
            y=alt.Y("volume_kwh:Q", title=f"Energy traded ({theme.KWH})"),
            tooltip=[
                alt.Tooltip("tick:Q", title="Tick"),
                alt.Tooltip("volume_kwh:Q", title=f"Volume ({theme.KWH})", format=".3f"),
            ],
        )
    )
    st.altair_chart(theme.style(chart, height=200), width="stretch")


def _outcomes_table(run: RunData) -> None:
    """Collapsed table view of the same numbers, for accessibility and checking."""
    with st.expander("Per-household outcomes (table)"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Household": o.household_id,
                    f"P2P cost ({theme.EUR})": round(o.total_cost_eur_p2p, 2),
                    f"Grid-only cost ({theme.EUR})": round(o.total_cost_eur_grid_baseline, 2),
                    f"Saved ({theme.EUR})": round(o.savings_eur, 2),
                    "Saved (%)": round(o.savings_pct, 2),
                }
                for o in run.outcomes
            ),
            hide_index=True,
            width="stretch",
        )


def _tariff_band(run: RunData) -> tuple[float, float]:
    """Highest import and lowest export tariff in the run, in EUR/kWh."""
    if not run.tariffs_eur_per_kwh:
        return tariff_for({}, "")
    imports = [i for i, _ in run.tariffs_eur_per_kwh.values()]
    exports = [e for _, e in run.tariffs_eur_per_kwh.values()]
    return max(imports), min(exports)
