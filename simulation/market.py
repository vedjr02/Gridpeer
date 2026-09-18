"""Continuous double auction: turn a tick's orders into cleared trades.

Build-order step 1/3 (see simulation/CLAUDE.md). This module owns the market
mechanism and nothing else — it does not decide what any household bids, it only
clears whatever orders it is given, and it never reaches into another module's
internals. The only things crossing this boundary are shared contracts:
``AgentDecision`` in, ``TradeEvent``/``MarketState`` out.

Mechanism, in one paragraph: buy orders are ranked by descending limit price and
sell orders by ascending limit price, ties broken by arrival order in the input
sequence (price-time priority). The best buy is matched against the best sell for
as long as the buyer's maximum price is at least the seller's minimum price. Each
match trades the smaller of the two quantities; the larger order keeps its
remainder on the book and can match again. When the best buy no longer crosses
the best sell, nothing further can trade and the rest is unmatched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from shared.schemas import AgentDecision, MarketState, OrderSide, TradeEvent

# Quantities are floats, so a "fully filled" order rarely lands exactly on zero.
# Anything below this (1 µWh) is treated as filled — well under the resolution of
# any real meter reading, and it stops float dust from becoming a phantom order.
QUANTITY_EPSILON_KWH = 1e-9


@dataclass
class _Entry:
    """One order resting on the book, with however much of it is still open.

    ``arrival`` is the order's position in the input sequence — the "time" half of
    price-time priority. Orders within a tick arrive together, so position is the
    only ordering available; holding it explicitly keeps clearing reproducible
    instead of dependent on sort stability.
    """

    arrival: int
    order: AgentDecision
    remaining_kwh: float

    @property
    def filled(self) -> bool:
        """True once nothing tradeable is left (float dust below the epsilon)."""
        return self.remaining_kwh <= QUANTITY_EPSILON_KWH


def trade_id(tick: int, sequence: int) -> str:
    """Deterministic id for the ``sequence``-th trade cleared in ``tick``.

    Deterministic on purpose: reproducing a specific run is how the agents module
    debugs why a policy behaved the way it did on a given tick.
    """
    return f"t_{tick:04d}_{sequence:03d}"


def _clearing_price_eur_per_kwh(buy: AgentDecision, sell: AgentDecision) -> float:
    """Price for one match: the midpoint of the two limit prices.

    The midpoint splits the surplus evenly between buyer and seller. Any price in
    the crossing range would clear, but a rule that systematically favoured one
    side would bias the savings comparison the whole project is built to make —
    and would quietly teach an RL agent to exploit the mechanism instead of
    learning to trade.
    """
    return (buy.limit_price_eur_per_kwh + sell.limit_price_eur_per_kwh) / 2


def _validate(tick: int, orders: Sequence[AgentDecision]) -> None:
    """Reject order books that violate the one-order-per-household-per-tick rule."""
    seen: set[str] = set()
    for order in orders:
        if order.tick != tick:
            raise ValueError(
                f"order from {order.household_id} has tick {order.tick}, "
                f"but this book is clearing tick {tick}"
            )
        if order.household_id in seen:
            raise ValueError(
                f"household {order.household_id} submitted more than one order for "
                f"tick {tick}; the contract is one AgentDecision per household per tick"
            )
        seen.add(order.household_id)


def clear_tick(
    tick: int,
    timestamp: datetime,
    orders: Sequence[AgentDecision],
) -> MarketState:
    """Clear one tick's order book and return the resulting market state.

    Orders are matched by price-time priority (see module docstring). Unmatched
    orders — including the unfilled remainder of a partially filled one — are
    returned in ``MarketState`` so the caller can settle them against the
    household's grid import/export tariff instead. That fallback is what makes the
    P2P-versus-grid-only comparison meaningful.

    ``MarketState.clearing_price_eur_per_kwh`` is the volume-weighted average price
    across the tick's trades, or ``None`` when nothing cleared.
    """
    _validate(tick, orders)

    buys = [
        _Entry(index, order, order.quantity_kwh)
        for index, order in enumerate(orders)
        if order.side is OrderSide.BUY
    ]
    sells = [
        _Entry(index, order, order.quantity_kwh)
        for index, order in enumerate(orders)
        if order.side is OrderSide.SELL
    ]
    buys.sort(key=lambda entry: (-entry.order.limit_price_eur_per_kwh, entry.arrival))
    sells.sort(key=lambda entry: (entry.order.limit_price_eur_per_kwh, entry.arrival))

    trades: list[TradeEvent] = []
    buy_cursor = 0
    sell_cursor = 0

    while buy_cursor < len(buys) and sell_cursor < len(sells):
        best_buy = buys[buy_cursor]
        best_sell = sells[sell_cursor]
        buy_order = best_buy.order
        sell_order = best_sell.order

        # Best buy no longer meets best ask: no remaining pair can cross either,
        # because every later buy is cheaper and every later sell dearer.
        if buy_order.limit_price_eur_per_kwh < sell_order.limit_price_eur_per_kwh:
            break

        quantity_kwh = min(best_buy.remaining_kwh, best_sell.remaining_kwh)
        trades.append(
            TradeEvent(
                trade_id=trade_id(tick, len(trades)),
                tick=tick,
                timestamp=timestamp,
                buyer_id=buy_order.household_id,
                seller_id=sell_order.household_id,
                quantity_kwh=quantity_kwh,
                clearing_price_eur_per_kwh=_clearing_price_eur_per_kwh(buy_order, sell_order),
            )
        )

        best_buy.remaining_kwh -= quantity_kwh
        best_sell.remaining_kwh -= quantity_kwh
        if best_buy.filled:
            buy_cursor += 1
        if best_sell.filled:
            sell_cursor += 1

    return MarketState(
        tick=tick,
        timestamp=timestamp,
        trades=trades,
        unmatched_buy_orders=_unmatched(buys, buy_cursor),
        unmatched_sell_orders=_unmatched(sells, sell_cursor),
        clearing_price_eur_per_kwh=_volume_weighted_price(trades),
    )


def _unmatched(book: list[_Entry], cursor: int) -> list[AgentDecision]:
    """Orders still open after clearing, in original arrival order.

    A partially filled order comes back with its *remaining* quantity, so the
    caller settles only the part the market could not fill. Fully filled orders
    are dropped rather than returned at quantity zero — ``AgentDecision``
    requires a positive quantity, and "nothing left to trade" is absence, not a
    zero-sized order.
    """
    still_open = sorted(
        (entry for entry in book[cursor:] if not entry.filled),
        key=lambda entry: entry.arrival,
    )
    return [
        entry.order
        if entry.remaining_kwh == entry.order.quantity_kwh
        else entry.order.model_copy(update={"quantity_kwh": entry.remaining_kwh})
        for entry in still_open
    ]


def _volume_weighted_price(trades: Sequence[TradeEvent]) -> float | None:
    """Volume-weighted average clearing price for a tick, or None if nothing cleared.

    Volume-weighted rather than a plain mean: a 10 kWh trade should move the tick's
    headline price more than a 0.1 kWh one, and the dashboard plots this as *the*
    price for the tick.
    """
    if not trades:
        return None
    total_kwh = sum(trade.quantity_kwh for trade in trades)
    if total_kwh <= 0:
        return None
    return (
        sum(trade.clearing_price_eur_per_kwh * trade.quantity_kwh for trade in trades) / total_kwh
    )
