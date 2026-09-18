"""Tests for the continuous double auction.

The two-household case below is the hand-calculated sanity check simulation/CLAUDE.md
asks for: it stays in the suite permanently, and if it ever breaks, something
fundamental about the mechanism changed.
"""

from datetime import datetime

import pytest

from shared.schemas import AgentDecision, MarketState, OrderSide, TradeEvent
from simulation.market import clear_tick, trade_id

TICK = 1
TIMESTAMP = datetime(2026, 1, 1, 8, 0)


def order(
    household_id: str,
    side: OrderSide,
    quantity_kwh: float,
    limit_price_eur_per_kwh: float,
    tick: int = TICK,
) -> AgentDecision:
    """An AgentDecision for the book, with the fields these tests don't vary fixed."""
    return AgentDecision(
        household_id=household_id,
        tick=tick,
        side=side,
        quantity_kwh=quantity_kwh,
        limit_price_eur_per_kwh=limit_price_eur_per_kwh,
        strategy_name="test_fixture",
    )


# ---------------------------------------------------------------------------
# The permanent hand-calculated case
# ---------------------------------------------------------------------------

def test_two_households_clear_at_hand_calculated_price():
    """hh_002 will pay up to 0.20, hh_001 will sell from 0.10, both want 2.0 kWh.

    Worked by hand: the orders cross (0.20 >= 0.10), so the whole 2.0 kWh trades at
    the midpoint, (0.20 + 0.10) / 2 = 0.15 EUR/kWh. Nothing is left over on either
    side, so the tick's price is that single trade's price.
    """
    state = clear_tick(
        tick=TICK,
        timestamp=TIMESTAMP,
        orders=[
            order("hh_001", OrderSide.SELL, quantity_kwh=2.0, limit_price_eur_per_kwh=0.10),
            order("hh_002", OrderSide.BUY, quantity_kwh=2.0, limit_price_eur_per_kwh=0.20),
        ],
    )

    assert len(state.trades) == 1
    trade = state.trades[0]
    assert trade.buyer_id == "hh_002"
    assert trade.seller_id == "hh_001"
    assert trade.quantity_kwh == 2.0
    # approx, not ==: the midpoint of two floats lands a bit-width away from 0.15.
    # The engine deliberately does not round — how money is rounded is a settlement
    # decision for the team, not something the matching loop should quietly impose.
    assert trade.clearing_price_eur_per_kwh == pytest.approx(0.15)
    assert state.clearing_price_eur_per_kwh == pytest.approx(0.15)
    assert state.unmatched_buy_orders == []
    assert state.unmatched_sell_orders == []


# ---------------------------------------------------------------------------
# No cross
# ---------------------------------------------------------------------------

def test_no_cross_leaves_everything_unmatched_and_price_none():
    """A buyer capped below the seller's floor must not trade.

    This is the path that hands both households to the grid-tariff fallback, so
    "no trade" has to be represented cleanly rather than as a zero-priced trade.
    """
    buy = order("hh_002", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.08)
    sell = order("hh_001", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.12)

    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=[buy, sell])

    assert state.trades == []
    assert state.clearing_price_eur_per_kwh is None
    assert state.unmatched_buy_orders == [buy]
    assert state.unmatched_sell_orders == [sell]


def test_empty_book_clears_to_an_empty_market_state():
    """A tick where nobody bid is legal, not an error."""
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=[])

    assert state.trades == []
    assert state.clearing_price_eur_per_kwh is None
    assert state.unmatched_buy_orders == []
    assert state.unmatched_sell_orders == []


def test_exactly_equal_prices_still_cross():
    """Buyer's maximum equal to the seller's minimum is a match, not a miss."""
    state = clear_tick(
        tick=TICK,
        timestamp=TIMESTAMP,
        orders=[
            order("hh_001", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.15),
            order("hh_002", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.15),
        ],
    )

    assert len(state.trades) == 1
    assert state.trades[0].clearing_price_eur_per_kwh == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Partial fills
# ---------------------------------------------------------------------------

def test_partial_fill_returns_only_the_remainder_as_unmatched():
    """A 3.0 kWh buyer meeting a 1.0 kWh seller trades 1.0 and keeps 2.0 open."""
    state = clear_tick(
        tick=TICK,
        timestamp=TIMESTAMP,
        orders=[
            order("hh_002", OrderSide.BUY, quantity_kwh=3.0, limit_price_eur_per_kwh=0.20),
            order("hh_001", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.10),
        ],
    )

    assert len(state.trades) == 1
    assert state.trades[0].quantity_kwh == 1.0
    assert state.unmatched_sell_orders == []

    assert len(state.unmatched_buy_orders) == 1
    remainder = state.unmatched_buy_orders[0]
    assert remainder.household_id == "hh_002"
    assert remainder.quantity_kwh == 2.0
    # Everything except the quantity survives the partial fill untouched.
    assert remainder.limit_price_eur_per_kwh == pytest.approx(0.20)
    assert remainder.strategy_name == "test_fixture"


def test_one_seller_fills_several_buyers_best_price_first():
    """A large seller is consumed by buyers in descending price order."""
    state = clear_tick(
        tick=TICK,
        timestamp=TIMESTAMP,
        orders=[
            order("hh_low", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.14),
            order("hh_high", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.22),
            order("hh_seller", OrderSide.SELL, quantity_kwh=1.5, limit_price_eur_per_kwh=0.10),
        ],
    )

    assert [t.buyer_id for t in state.trades] == ["hh_high", "hh_low"]
    assert [t.quantity_kwh for t in state.trades] == [1.0, 0.5]
    # Prices: (0.22 + 0.10)/2 = 0.16, then (0.14 + 0.10)/2 = 0.12.
    assert [t.clearing_price_eur_per_kwh for t in state.trades] == pytest.approx([0.16, 0.12])
    # Volume-weighted: (0.16*1.0 + 0.12*0.5) / 1.5.
    assert state.clearing_price_eur_per_kwh == pytest.approx((0.16 * 1.0 + 0.12 * 0.5) / 1.5)
    assert state.unmatched_sell_orders == []
    assert state.unmatched_buy_orders[0].quantity_kwh == 0.5


# ---------------------------------------------------------------------------
# Price-time priority
# ---------------------------------------------------------------------------

def test_equal_prices_are_broken_by_arrival_order():
    """Two sellers at the same price: the one that arrived first fills first."""
    state = clear_tick(
        tick=TICK,
        timestamp=TIMESTAMP,
        orders=[
            order("hh_early", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.10),
            order("hh_late", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.10),
            order("hh_buyer", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.20),
        ],
    )

    assert [t.seller_id for t in state.trades] == ["hh_early"]
    assert [o.household_id for o in state.unmatched_sell_orders] == ["hh_late"]


def test_cheaper_seller_is_matched_before_a_dearer_one():
    """Price beats arrival order: a later, cheaper ask still goes first."""
    state = clear_tick(
        tick=TICK,
        timestamp=TIMESTAMP,
        orders=[
            order("hh_dear", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.18),
            order("hh_cheap", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.11),
            order("hh_buyer", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.20),
        ],
    )

    assert [t.seller_id for t in state.trades] == ["hh_cheap"]


def test_unmatched_orders_come_back_in_arrival_order():
    """Unmatched books are returned in submission order, not price order.

    The caller settles these against grid tariffs household by household; a stable,
    submission-ordered list keeps that loop (and any diff of it) reproducible.
    """
    state = clear_tick(
        tick=TICK,
        timestamp=TIMESTAMP,
        orders=[
            order("hh_a", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.05),
            order("hh_b", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.09),
            order("hh_c", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.07),
        ],
    )

    assert [o.household_id for o in state.unmatched_buy_orders] == ["hh_a", "hh_b", "hh_c"]


# ---------------------------------------------------------------------------
# Determinism and the shared contract
# ---------------------------------------------------------------------------

def test_clearing_is_deterministic_for_the_same_book():
    """Same orders in, byte-identical market state out — twice.

    Reproducibility is a hard requirement from simulation/CLAUDE.md: the agents
    module needs to replay a specific tick to debug a policy.
    """
    book = [
        order("hh_001", OrderSide.SELL, quantity_kwh=2.0, limit_price_eur_per_kwh=0.10),
        order("hh_002", OrderSide.BUY, quantity_kwh=1.5, limit_price_eur_per_kwh=0.20),
        order("hh_003", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.12),
    ]

    first = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=book)
    second = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=book)

    assert first.model_dump() == second.model_dump()


def test_trade_ids_are_unique_and_deterministic_within_a_tick():
    state = clear_tick(
        tick=7,
        timestamp=TIMESTAMP,
        orders=[
            order("hh_001", OrderSide.SELL, quantity_kwh=2.0, limit_price_eur_per_kwh=0.10, tick=7),
            order("hh_002", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.20, tick=7),
            order("hh_003", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.18, tick=7),
        ],
    )

    ids = [t.trade_id for t in state.trades]
    assert ids == [trade_id(7, 0), trade_id(7, 1)] == ["t_0007_000", "t_0007_001"]
    assert len(set(ids)) == len(ids)


def test_output_satisfies_the_shared_contract():
    """Every TradeEvent and the MarketState itself must round-trip through shared/."""
    state = clear_tick(
        tick=TICK,
        timestamp=TIMESTAMP,
        orders=[
            order("hh_001", OrderSide.SELL, quantity_kwh=2.0, limit_price_eur_per_kwh=0.10),
            order("hh_002", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.20),
        ],
    )

    assert MarketState.model_validate(state.model_dump()) == state
    for trade in state.trades:
        assert TradeEvent.model_validate(trade.model_dump()) == trade


# ---------------------------------------------------------------------------
# Rejected books
# ---------------------------------------------------------------------------

def test_order_from_another_tick_is_rejected():
    with pytest.raises(ValueError, match="tick"):
        clear_tick(
            tick=TICK,
            timestamp=TIMESTAMP,
            orders=[
                order(
                    "hh_001",
                    OrderSide.SELL,
                    quantity_kwh=1.0,
                    limit_price_eur_per_kwh=0.10,
                    tick=99,
                ),
            ],
        )


def test_two_orders_from_the_same_household_are_rejected():
    """One AgentDecision per household per tick — a second one is a caller bug."""
    with pytest.raises(ValueError, match="more than one order"):
        clear_tick(
            tick=TICK,
            timestamp=TIMESTAMP,
            orders=[
                order("hh_001", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.10),
                order("hh_001", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.20),
            ],
        )
