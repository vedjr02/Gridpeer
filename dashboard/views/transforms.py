"""Pure transforms from shared contracts to tidy frames the charts encode.

No Streamlit and no chart objects here — these are ordinary functions over
``HouseholdOutcome`` / ``TradeEvent`` / ``MarketState`` lists, so they can be
reasoned about (and tested) without a browser. Every returned column name
carries its unit.

These functions reshape; they do not re-derive. ``dashboard/outcomes.py`` owns
the savings arithmetic and its docstring is explicit that recomputing it
elsewhere is how two versions of the headline number start to disagree — so the
savings frame below reads ``HouseholdOutcome`` rather than re-deriving costs
from trades.
"""

from __future__ import annotations

import pandas as pd

from shared.schemas import HouseholdOutcome, MarketState, TradeEvent

#: Labels for the two cost bases, used as the colour domain on the savings chart.
P2P_BASIS = "Peer-to-peer"
GRID_BASIS = "Grid-only baseline"


def savings_by_household(outcomes: list[HouseholdOutcome]) -> pd.DataFrame:
    """Long frame of each household's run-long cost under both bases, in EUR.

    Cost is positive when money leaves the household (the convention settlement
    and outcomes.py use), so a net producer's cost is negative — that is revenue,
    not an error.
    """
    if not outcomes:
        return pd.DataFrame(columns=["household_id", "basis", "cost_eur", "savings_eur"])
    rows = [
        {
            "household_id": outcome.household_id,
            "basis": basis,
            "cost_eur": cost,
            "savings_eur": outcome.savings_eur,
            "savings_pct": outcome.savings_pct,
        }
        for outcome in outcomes
        for basis, cost in (
            (GRID_BASIS, outcome.total_cost_eur_grid_baseline),
            (P2P_BASIS, outcome.total_cost_eur_p2p),
        )
    ]
    return pd.DataFrame(rows)


def volume_per_tick(
    trades: list[TradeEvent], num_ticks: int, household_ids: list[str] | None = None
) -> pd.DataFrame:
    """Total energy cleared per tick, in kWh, zero-filled across the whole run."""
    index = pd.RangeIndex(num_ticks, name="tick")
    if not trades:
        return pd.DataFrame({"tick": index, "volume_kwh": 0.0})
    frame = pd.DataFrame({"tick": t.tick, "volume_kwh": t.quantity_kwh} for t in trades)
    totals = frame.groupby("tick")["volume_kwh"].sum().reindex(index, fill_value=0.0)
    return totals.reset_index()


def clearing_price_per_tick(market_states: list[MarketState]) -> pd.DataFrame:
    """Clearing price per tick in EUR/kWh; ticks that cleared nothing are dropped.

    A tick with no trades has no price — that is a genuine gap in the series, not
    a zero, and drawing it as zero would invent a price crash that never happened.
    """
    rows = [
        {
            "tick": state.tick,
            "clearing_price_eur_per_kwh": state.clearing_price_eur_per_kwh,
        }
        for state in market_states
        if state.clearing_price_eur_per_kwh is not None
    ]
    if not rows:
        return pd.DataFrame(columns=["tick", "clearing_price_eur_per_kwh"])
    return pd.DataFrame(rows).sort_values("tick").reset_index(drop=True)
