"""GridPeer dashboard (Streamlit) — the outcomes narrative.

This app exists to defend two claims, in this order:

    1. Households trading peer-to-peer saved X% versus just exporting to the grid.
    2. RL agents beat the rule-based baseline by Y%.

The Outcomes tab carries claim 1, Strategy comparison carries claim 2, and
anything supporting neither — forecast diagnostics, raw frames — lives in
Forecast detail. Every number on screen was persisted by a run; this module
only picks a run and routes to a view.

Run:  streamlit run dashboard/app.py
Point at a different database with GRIDPEER_DB=path/to/run.db
"""

from __future__ import annotations

import streamlit as st

from dashboard.views import comparison, data, forecast_detail, overview, theme
from shared.schemas import RunSummary


def main() -> None:
    """Render the sidebar run selector and the three dashboard tabs."""
    st.set_page_config(page_title="GridPeer", page_icon="⚡", layout="wide")
    st.title("⚡ GridPeer — peer-to-peer energy trading outcomes")

    store, is_fixture = data.open_store()
    summaries = data.list_summaries(store)
    selected = _run_selector(summaries, is_fixture)

    outcomes_tab, comparison_tab, forecast_tab = st.tabs(
        ["Outcomes", "Strategy comparison", "Forecast detail"]
    )
    run = None if selected is None else data.load_run(store, selected)

    with outcomes_tab:
        if run is None:
            st.info(
                "No summarised runs in this database yet. Run the orchestrator with a "
                "`db_path` to persist one, then reload this page."
            )
        else:
            overview.render(run)
    with comparison_tab:
        comparison.render(summaries)
    with forecast_tab:
        if run is None:
            st.info("Select a run to see its forecast stream.")
        else:
            forecast_detail.render(
                run.forecasts, theme.HouseholdPalette.for_run(run.household_ids)
            )


def _run_selector(summaries: list[RunSummary], is_fixture: bool) -> RunSummary | None:
    """Sidebar picker over persisted runs; None when nothing is stored yet."""
    with st.sidebar:
        st.header("Run")
        if is_fixture:
            st.warning(
                "Showing stand-in data — dashboard/persistence.py is not importable, "
                "so these numbers are shaped like real ones but mean nothing.",
                icon="⚠️",
            )
        if not summaries:
            st.caption(f"No runs found in `{data.DB_PATH}`.")
            return None

        by_id = {s.run_id: s for s in summaries}
        run_id = st.selectbox(
            "Select a run",
            list(by_id),
            format_func=lambda r: f"{by_id[r].strategy_name} · {r}",
        )
        selected = by_id[run_id]
        st.caption(
            f"Strategy: **{selected.strategy_name}**  \n"
            f"{selected.num_households} households · {selected.num_ticks} ticks"
        )
        return selected


if __name__ == "__main__":
    main()
