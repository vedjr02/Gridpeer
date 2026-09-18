"""Strategy comparison tab: the week-3 money shot, framed now.

The project's definition of done is one defensible sentence — "our RL agents
produced Y% more household savings than the rule-based baseline." This tab is
where that sentence gets its number. The RunSummary diff below is real and works
today against any two persisted runs; the per-household and significance layers
are deliberately left as TODOs until RL runs exist to compare.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.views import theme
from shared.schemas import RunSummary


def render(summaries: list[RunSummary]) -> None:
    """Pick two runs and diff their RunSummary side by side."""
    st.subheader("Strategy comparison")
    st.caption(
        "Baseline versus learned strategy on the same households and ticks. "
        "The uplift below is the number the project is ultimately judged on."
    )

    if len(summaries) < 2:
        st.info(
            "Two persisted runs are needed to compare — run the baseline strategy and "
            "an RL strategy, then select one of each here."
        )
        return

    by_id = {s.run_id: s for s in summaries}
    labels = {s.run_id: f"{s.strategy_name} · {s.run_id}" for s in summaries}
    run_ids = list(by_id)

    col1, col2 = st.columns(2)
    baseline_id = col1.selectbox(
        "Baseline run", run_ids, index=0, format_func=lambda r: labels[r]
    )
    candidate_id = col2.selectbox(
        "Candidate run", run_ids, index=1, format_func=lambda r: labels[r]
    )

    if baseline_id == candidate_id:
        st.warning("Pick two different runs to see a difference.")
        return

    _uplift_metrics(by_id[baseline_id], by_id[candidate_id])
    st.divider()
    _diff_table(by_id[baseline_id], by_id[candidate_id])
    _pending_work()


def _uplift_metrics(baseline: RunSummary, candidate: RunSummary) -> None:
    """Headline deltas, each on its own scale — never two units on one axis."""
    st.markdown(
        f"**{candidate.strategy_name}** vs **{baseline.strategy_name}**"
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        f"Total savings ({theme.EUR})",
        f"{candidate.total_savings_eur:,.2f}",
        delta=f"{_pct_change(baseline.total_savings_eur, candidate.total_savings_eur):+.1f}%",
        help="The project's headline claim: more household savings than the baseline.",
    )
    col2.metric(
        "Avg savings (%)",
        f"{candidate.avg_savings_pct:.1f}%",
        delta=f"{candidate.avg_savings_pct - baseline.avg_savings_pct:+.1f} pp",
    )
    col3.metric(
        "CO2 avoided (kg)",
        f"{candidate.co2_avoided_kg:,.1f}",
        delta=f"{_pct_change(baseline.co2_avoided_kg, candidate.co2_avoided_kg):+.1f}%",
    )
    col4.metric(
        "Peak-load reduction (%)",
        f"{candidate.peak_load_reduction_pct:.1f}%",
        delta=(
            f"{candidate.peak_load_reduction_pct - baseline.peak_load_reduction_pct:+.1f} pp"
        ),
    )

    if baseline.num_households != candidate.num_households or (
        baseline.num_ticks != candidate.num_ticks
    ):
        st.warning(
            "These runs cover different households or tick counts — the uplift is "
            "not a like-for-like comparison.",
            icon="⚠️",
        )


def _diff_table(baseline: RunSummary, candidate: RunSummary) -> None:
    """Every RunSummary field side by side, with the change between them."""
    rows = [
        ("Strategy", baseline.strategy_name, candidate.strategy_name, ""),
        ("Households", f"{baseline.num_households}", f"{candidate.num_households}", ""),
        ("Ticks", f"{baseline.num_ticks}", f"{candidate.num_ticks}", ""),
        (
            f"Total savings ({theme.EUR})",
            f"{baseline.total_savings_eur:,.2f}",
            f"{candidate.total_savings_eur:,.2f}",
            f"{_pct_change(baseline.total_savings_eur, candidate.total_savings_eur):+.1f}%",
        ),
        (
            "Avg savings (%)",
            f"{baseline.avg_savings_pct:.2f}",
            f"{candidate.avg_savings_pct:.2f}",
            f"{candidate.avg_savings_pct - baseline.avg_savings_pct:+.2f} pp",
        ),
        (
            "CO2 avoided (kg)",
            f"{baseline.co2_avoided_kg:,.2f}",
            f"{candidate.co2_avoided_kg:,.2f}",
            f"{_pct_change(baseline.co2_avoided_kg, candidate.co2_avoided_kg):+.1f}%",
        ),
        (
            "Peak-load reduction (%)",
            f"{baseline.peak_load_reduction_pct:.2f}",
            f"{candidate.peak_load_reduction_pct:.2f}",
            f"{candidate.peak_load_reduction_pct - baseline.peak_load_reduction_pct:+.2f} pp",
        ),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Metric", "Baseline", "Candidate", "Change"]),
        hide_index=True,
        width="stretch",
    )


def _pending_work() -> None:
    """State what this tab still owes, so the gap is visible rather than assumed."""
    with st.expander("Not built yet"):
        st.markdown(
            "- Per-household savings diff — which households the learned strategy "
            "helps, and which it does not.\n"
            "- Repeated runs per strategy, so the uplift comes with a spread rather "
            "than a single number.\n"
            "- Clearing-price and volume overlay for both runs on shared axes."
        )


def _pct_change(baseline_value: float, candidate_value: float) -> float:
    """Percentage change from baseline to candidate; 0.0 when the baseline is zero."""
    if baseline_value == 0:
        return 0.0
    return 100.0 * (candidate_value - baseline_value) / abs(baseline_value)
