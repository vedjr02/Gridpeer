"""Property-based tests: rules the market must obey for *every* order book.

The example-based tests in test_market.py check books someone thought of. These
generate hundreds of random books per run — many households, both sides, prices
drawn from a coarse grid so ties are common — and check invariants that no book may
break. When one fails, Hypothesis shrinks it to the smallest book that still breaks
it, which is usually the whole bug report.

The invariants, and why each matters to the project:

- **Energy balances** — every order's traded plus unmatched volume is exactly what
  it offered; nothing is created or lost in clearing.
- **Nobody trades at a price they refused** — for both pricing rules.
- **The market is cleared** — once clearing stops, no remaining buy and sell cross.
- **Book order never matters** — shuffling the input changes nothing, so the first
  household in a list has no advantage (the bug the pro-rata rule fixed).
- **Ties are shared pro-rata** — orders at one price on the long side all fill the
  same fraction.
- **Pricing rules change prices only** — both rules agree on who trades how much.
- **Honest trading never loses money** — settled at the meter, an order that matches
  the household's real position, priced inside the tariff gap, saves or breaks even.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shared.schemas import AgentDecision, AgentRole, HouseholdProfile, MarketState, OrderSide
from simulation.market import QUANTITY_EPSILON_KWH, PricingRule, clear_tick
from simulation.settlement import settle_tick

TICK = 3
TIMESTAMP = datetime(2026, 1, 1, 12, 0)
IMPORT_TARIFF = 0.25
EXPORT_TARIFF = 0.07

# A coarse grid, so many generated orders share a price and the tie rule is exercised
# constantly rather than by luck.
PRICE_GRID_EUR_PER_KWH = [round(0.05 + 0.01 * step, 2) for step in range(21)]
TARIFF_GAP_PRICES = [
    price for price in PRICE_GRID_EUR_PER_KWH if EXPORT_TARIFF <= price <= IMPORT_TARIFF
]

# Loose enough for float accumulation across a few dozen partial fills, far below
# anything a meter or a tariff could resolve.
TOLERANCE = 1e-7

# 200 books per property keeps the suite fast in CI; HYPOTHESIS_PROFILE=thorough runs
# 5,000 for a deeper local check before changing the mechanism.
settings.register_profile("gridpeer", deadline=None, max_examples=200)
settings.register_profile("thorough", deadline=None, max_examples=5_000)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "gridpeer"))


@st.composite
def order_books(draw, prices=PRICE_GRID_EUR_PER_KWH, max_households=14):
    """A tick's book: up to ``max_households`` households, one order each."""
    size = draw(st.integers(min_value=0, max_value=max_households))
    return [
        AgentDecision(
            household_id=f"hh_{index:02d}",
            tick=TICK,
            side=draw(st.sampled_from([OrderSide.BUY, OrderSide.SELL])),
            quantity_kwh=draw(st.floats(min_value=0.01, max_value=10.0, allow_nan=False)),
            limit_price_eur_per_kwh=draw(st.sampled_from(prices)),
            strategy_name="property_test",
        )
        for index in range(size)
    ]


pricing_rules = st.sampled_from(list(PricingRule))


def traded_kwh_by_household(state: MarketState) -> dict[str, float]:
    """Energy each household bought or sold this tick."""
    traded: dict[str, float] = defaultdict(float)
    for trade in state.trades:
        traded[trade.buyer_id] += trade.quantity_kwh
        traded[trade.seller_id] += trade.quantity_kwh
    return traded


def unmatched_kwh_by_household(state: MarketState) -> dict[str, float]:
    """Energy each household offered but did not trade."""
    unmatched = state.unmatched_buy_orders + state.unmatched_sell_orders
    return {order.household_id: order.quantity_kwh for order in unmatched}


@given(book=order_books(), pricing=pricing_rules)
def test_every_order_is_traded_or_returned_in_full(book, pricing):
    """Traded + unmatched == offered, for every order: clearing conserves energy."""
    state = clear_tick(TICK, TIMESTAMP, book, pricing=pricing)
    traded = traded_kwh_by_household(state)
    unmatched = unmatched_kwh_by_household(state)

    for order in book:
        accounted = traded.get(order.household_id, 0.0) + unmatched.get(order.household_id, 0.0)
        assert accounted == pytest.approx(order.quantity_kwh, abs=TOLERANCE)


@given(book=order_books(), pricing=pricing_rules)
def test_households_only_trade_on_the_side_they_asked_for(book, pricing):
    """A buyer never sells and a seller never buys; nobody trades with themselves."""
    state = clear_tick(TICK, TIMESTAMP, book, pricing=pricing)
    sides = {order.household_id: order.side for order in book}

    for trade in state.trades:
        assert sides[trade.buyer_id] is OrderSide.BUY
        assert sides[trade.seller_id] is OrderSide.SELL
        assert trade.buyer_id != trade.seller_id
        assert trade.quantity_kwh > QUANTITY_EPSILON_KWH


@given(book=order_books(), pricing=pricing_rules)
def test_nobody_trades_at_a_price_they_refused(book, pricing):
    """Every trade price is at or below the buyer's limit and at or above the seller's."""
    state = clear_tick(TICK, TIMESTAMP, book, pricing=pricing)
    limits = {order.household_id: order.limit_price_eur_per_kwh for order in book}

    for trade in state.trades:
        assert limits[trade.seller_id] - TOLERANCE <= trade.clearing_price_eur_per_kwh
        assert trade.clearing_price_eur_per_kwh <= limits[trade.buyer_id] + TOLERANCE


@given(book=order_books(), pricing=pricing_rules)
def test_nothing_that_could_trade_is_left_on_the_book(book, pricing):
    """After clearing, the best remaining buy is strictly below the best remaining sell."""
    state = clear_tick(TICK, TIMESTAMP, book, pricing=pricing)

    if state.unmatched_buy_orders and state.unmatched_sell_orders:
        best_buy = max(o.limit_price_eur_per_kwh for o in state.unmatched_buy_orders)
        best_sell = min(o.limit_price_eur_per_kwh for o in state.unmatched_sell_orders)
        assert best_buy < best_sell


@given(data=st.data(), book=order_books(), pricing=pricing_rules)
def test_shuffling_the_book_changes_nothing(data, book, pricing):
    """Same orders in any order, same trades — ids included. No first-in-list advantage."""
    shuffled = data.draw(st.permutations(book))

    original = clear_tick(TICK, TIMESTAMP, book, pricing=pricing)
    reordered = clear_tick(TICK, TIMESTAMP, shuffled, pricing=pricing)

    assert original.trades == reordered.trades
    assert unmatched_kwh_by_household(original) == unmatched_kwh_by_household(reordered)


@given(book=order_books(), pricing=pricing_rules)
def test_orders_tied_on_price_fill_the_same_fraction(book, pricing):
    """Within one side and one price, every order fills the same share of its size."""
    state = clear_tick(TICK, TIMESTAMP, book, pricing=pricing)
    traded = traded_kwh_by_household(state)

    fractions: dict[tuple[OrderSide, float], list[float]] = defaultdict(list)
    for order in book:
        fraction = traded.get(order.household_id, 0.0) / order.quantity_kwh
        fractions[(order.side, order.limit_price_eur_per_kwh)].append(fraction)

    for level in fractions.values():
        assert max(level) - min(level) <= TOLERANCE


@given(book=order_books())
def test_pricing_rules_agree_on_who_trades_how_much(book):
    """The two rules differ only in price, so they can be compared like for like."""
    pairwise = clear_tick(TICK, TIMESTAMP, book, pricing=PricingRule.PAIRWISE_MIDPOINT)
    uniform = clear_tick(TICK, TIMESTAMP, book, pricing=PricingRule.UNIFORM)

    assert [(t.buyer_id, t.seller_id, t.quantity_kwh) for t in pairwise.trades] == [
        (t.buyer_id, t.seller_id, t.quantity_kwh) for t in uniform.trades
    ]
    assert len({t.clearing_price_eur_per_kwh for t in uniform.trades}) <= 1


@given(book=order_books(prices=TARIFF_GAP_PRICES), pricing=pricing_rules)
def test_honest_trading_inside_the_tariff_gap_never_loses_money(book, pricing):
    """Orders that match the meter, priced between the tariffs, save or break even.

    Each household's meter reads exactly what it ordered — a seller has that surplus,
    a buyer that deficit — so there is no imbalance to settle at the grid, and every
    trade lands between what the grid would pay and charge. If this ever fails, the
    marketplace can make an honest household worse off than doing nothing, and the
    project's headline number is indefensible.
    """
    state = clear_tick(TICK, TIMESTAMP, book, pricing=pricing)
    profiles = {
        order.household_id: HouseholdProfile(
            household_id=order.household_id,
            role=AgentRole.PROSUMER,
            has_solar=True,
            solar_capacity_kw=3.0,
            battery_capacity_kwh=0.0,
            grid_export_tariff_eur_per_kwh=EXPORT_TARIFF,
            grid_import_tariff_eur_per_kwh=IMPORT_TARIFF,
        )
        for order in book
    }
    metered_kwh = {
        order.household_id: (
            order.quantity_kwh if order.side is OrderSide.SELL else -order.quantity_kwh
        )
        for order in book
    }

    settlements = settle_tick(state, profiles, actual_net_position_kwh=metered_kwh)

    for settlement in settlements.values():
        assert settlement.savings_eur >= -TOLERANCE
