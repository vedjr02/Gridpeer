"""Tests for the dashboard's frame transforms and colour assignment.

These cover the pure logic behind the charts — the part that can be wrong
silently. Streamlit rendering is not tested here; it is exercised by running
the app.
"""

from __future__ import annotations

from datetime import datetime

from dashboard.views import theme, transforms
from shared.schemas import HouseholdOutcome, MarketState, TradeEvent

_EPOCH = datetime(2026, 1, 1, 0, 0)


def _outcome(household_id: str, p2p: float, grid: float) -> HouseholdOutcome:
    """A HouseholdOutcome with savings derived the way outcomes.py derives it."""
    savings = grid - p2p
    reference = abs(grid)
    return HouseholdOutcome(
        household_id=household_id,
        total_cost_eur_p2p=p2p,
        total_cost_eur_grid_baseline=grid,
        savings_eur=savings,
        savings_pct=(savings / reference * 100.0) if reference > 0 else 0.0,
    )


def _trade(tick: int, quantity_kwh: float, price: float) -> TradeEvent:
    """One cleared trade at ``tick``."""
    return TradeEvent(
        trade_id=f"t{tick}-{quantity_kwh}",
        tick=tick,
        timestamp=_EPOCH,
        buyer_id="hh_001",
        seller_id="hh_000",
        quantity_kwh=quantity_kwh,
        clearing_price_eur_per_kwh=price,
    )


def test_savings_frame_carries_both_bases_for_every_household() -> None:
    frame = transforms.savings_by_household(
        [_outcome("hh_000", 8.0, 10.0), _outcome("hh_001", 5.0, 5.0)]
    )
    assert sorted(frame["household_id"].unique()) == ["hh_000", "hh_001"]
    assert sorted(frame["basis"].unique()) == sorted([transforms.GRID_BASIS, transforms.P2P_BASIS])
    assert len(frame) == 4


def test_savings_frame_reports_outcome_values_unchanged() -> None:
    """The chart must never re-derive savings — outcomes.py owns that number."""
    frame = transforms.savings_by_household([_outcome("hh_000", 8.0, 10.0)])
    assert frame["savings_eur"].unique().tolist() == [2.0]
    p2p = frame.loc[frame["basis"] == transforms.P2P_BASIS, "cost_eur"].item()
    grid = frame.loc[frame["basis"] == transforms.GRID_BASIS, "cost_eur"].item()
    assert (p2p, grid) == (8.0, 10.0)


def test_savings_frame_keeps_a_household_that_never_traded() -> None:
    frame = transforms.savings_by_household([_outcome("hh_002", 4.0, 4.0)])
    assert frame["household_id"].tolist() == ["hh_002", "hh_002"]
    assert frame["savings_eur"].unique().tolist() == [0.0]


def test_savings_frame_is_empty_without_outcomes() -> None:
    assert transforms.savings_by_household([]).empty


def test_volume_per_tick_zero_fills_ticks_that_cleared_nothing() -> None:
    frame = transforms.volume_per_tick([_trade(1, 2.0, 0.15)], num_ticks=4)
    assert frame["tick"].tolist() == [0, 1, 2, 3]
    assert frame["volume_kwh"].tolist() == [0.0, 2.0, 0.0, 0.0]


def test_volume_per_tick_sums_concurrent_trades() -> None:
    frame = transforms.volume_per_tick(
        [_trade(0, 1.5, 0.15), _trade(0, 0.5, 0.16)], num_ticks=1
    )
    assert frame["volume_kwh"].tolist() == [2.0]


def test_volume_per_tick_without_trades_is_all_zero() -> None:
    frame = transforms.volume_per_tick([], num_ticks=3)
    assert frame["volume_kwh"].tolist() == [0.0, 0.0, 0.0]


def test_clearing_price_drops_ticks_with_no_price() -> None:
    """A tick that cleared nothing is a gap, not a zero — zero invents a crash."""
    states = [
        MarketState(tick=0, timestamp=_EPOCH, trades=[], unmatched_buy_orders=[],
                    unmatched_sell_orders=[], clearing_price_eur_per_kwh=0.15),
        MarketState(tick=1, timestamp=_EPOCH, trades=[], unmatched_buy_orders=[],
                    unmatched_sell_orders=[], clearing_price_eur_per_kwh=None),
    ]
    frame = transforms.clearing_price_per_tick(states)
    assert frame["tick"].tolist() == [0]
    assert frame["clearing_price_eur_per_kwh"].tolist() == [0.15]


def test_clearing_price_is_empty_when_nothing_ever_cleared() -> None:
    frame = transforms.clearing_price_per_tick([])
    assert frame.empty
    assert list(frame.columns) == ["tick", "clearing_price_eur_per_kwh"]


def test_palette_does_not_repaint_when_a_chart_plots_a_subset() -> None:
    """Colour follows the household, not its position in the list a chart got."""
    palette = theme.HouseholdPalette.for_run(["hh_000", "hh_001", "hh_002"])

    # hh_001 never traded, so a chart built from trades only sees these two.
    plotted = ["hh_000", "hh_002"]
    assert [palette.colours[h] for h in plotted] == ["#2a78d6", "#1baf7a"]

    # The scale still spans the whole roster, so the legend and every other
    # chart in the run agree with it.
    scale = palette.scale()
    assert list(scale.domain) == ["hh_000", "hh_001", "hh_002"]
    assert dict(zip(scale.domain, scale.range, strict=True))["hh_002"] == palette.colours["hh_002"]


def test_palette_is_fixed_by_the_roster_not_by_call_order() -> None:
    """Two charts in the same run get identical colours whatever order they ask in."""
    roster = ["hh_002", "hh_000", "hh_001"]
    assert (
        theme.HouseholdPalette.for_run(roster).colours
        == theme.HouseholdPalette.for_run(sorted(roster)).colours
    )


def test_household_colours_are_distinct_until_the_palette_runs_out() -> None:
    colours = theme.household_colours([f"hh_{i:03d}" for i in range(8)])
    assert len(set(colours.values())) == 8


def test_ninth_household_folds_into_one_overflow_colour() -> None:
    """A ninth series is never a generated hue — it folds into a shared colour."""
    colours = theme.household_colours([f"hh_{i:03d}" for i in range(10)])
    assert colours["hh_008"] == colours["hh_009"]
    assert colours["hh_008"] not in {colours[f"hh_{i:03d}"] for i in range(8)}
