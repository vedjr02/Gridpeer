"""Per-tick double auction: turn a tick's orders into cleared trades.

Build-order step 1/3 (see simulation/CLAUDE.md). This module owns the market
mechanism and nothing else — it does not decide what any household bids, it only
clears whatever orders it is given, and it never reaches into another module's
internals. The only things crossing this boundary are shared contracts:
``AgentDecision`` in, ``TradeEvent``/``MarketState`` out.

What kind of auction this is, precisely: a *call* auction (a sealed-bid double
auction). Every household submits at most one order per half-hour tick, the whole
book is collected, and it is cleared once. It is not a continuous double auction —
there is no order flow within a tick, so orders never meet one at a time as they
arrive, and no order has an arrival time that could earn it priority.

Mechanism, in one paragraph: buy orders are grouped into price levels from the
highest limit down, sell orders from the lowest limit up. The best buy level is
matched against the best sell level for as long as the buyers' maximum is at least
the sellers' minimum. The volume that trades between two levels is the smaller of
their two totals, and the side with more on offer shares that volume **pro-rata**:
each order fills in proportion to its size. When the best buy level no longer
crosses the best sell level, nothing further can trade and the rest is unmatched.

Why pro-rata rather than first-come-first-served: all of a tick's orders arrive
together, so the only "first" available is an order's position in the input list —
which is an accident of how the caller happened to loop over households. Ranking by
it made the first household in the list win every tie, every tick, for a whole run.
Pro-rata is the standard exchange rule for simultaneous orders at one price, needs no
randomness, and gives the same result whatever order the book arrives in.
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

    ``arrival`` is the order's position in the input sequence. It never decides who
    trades (see module docstring); it only puts the unmatched orders back in the
    order they were submitted, so the caller's settlement loop stays reproducible.
    """

    arrival: int
    order: AgentDecision
    remaining_kwh: float

    @property
    def filled(self) -> bool:
        """True once nothing tradeable is left (float dust below the epsilon)."""
        return self.remaining_kwh <= QUANTITY_EPSILON_KWH


@dataclass(frozen=True)
class _Match:
    """Energy agreed between one buy and one sell order, before it is priced."""

    buy: _Entry
    sell: _Entry
    quantity_kwh: float


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


def _price_levels(book: list[_Entry], highest_first: bool) -> list[list[_Entry]]:
    """Group one side of the book into price levels, best level first.

    Orders tie only on exactly equal limit prices. Within a level they are ordered by
    household_id, never by input position, so the pairing below — and with it every
    trade id — does not depend on the order the caller submitted the book in.
    """
    direction = -1.0 if highest_first else 1.0
    ordered = sorted(
        book,
        key=lambda entry: (
            direction * entry.order.limit_price_eur_per_kwh,
            entry.order.household_id,
        ),
    )
    levels: list[list[_Entry]] = []
    for entry in ordered:
        if levels and (
            levels[-1][0].order.limit_price_eur_per_kwh == entry.order.limit_price_eur_per_kwh
        ):
            levels[-1].append(entry)
        else:
            levels.append([entry])
    return levels


def _pro_rata_kwh(level: list[_Entry], open_kwh: float, volume_kwh: float) -> list[float]:
    """How much of ``volume_kwh`` each order in a level fills, in proportion to its size.

    When the whole level trades, each order fills exactly what it has open; computing
    ``remaining * volume / open`` there would only add float error.
    """
    if volume_kwh >= open_kwh:
        return [entry.remaining_kwh for entry in level]
    share = volume_kwh / open_kwh
    return [entry.remaining_kwh * share for entry in level]


def _match_levels(buys: list[_Entry], sells: list[_Entry]) -> list[_Match]:
    """Trade two crossing price levels against each other, pro-rata within each.

    Each side's allocation is fixed first, then the two are paired off in
    household_id order. Pairing that way produces at most len(buys) + len(sells) - 1
    trades — never one per buyer-seller pair, which would be quadratic in a large
    community — while every household still gets exactly its pro-rata share.
    """
    open_buy_kwh = sum(entry.remaining_kwh for entry in buys)
    open_sell_kwh = sum(entry.remaining_kwh for entry in sells)
    volume_kwh = min(open_buy_kwh, open_sell_kwh)
    buy_allocation = _pro_rata_kwh(buys, open_buy_kwh, volume_kwh)
    sell_allocation = _pro_rata_kwh(sells, open_sell_kwh, volume_kwh)

    matches: list[_Match] = []
    b = s = 0
    while b < len(buys) and s < len(sells):
        quantity_kwh = min(buy_allocation[b], sell_allocation[s])
        if quantity_kwh > QUANTITY_EPSILON_KWH:
            matches.append(_Match(buys[b], sells[s], quantity_kwh))
            buys[b].remaining_kwh -= quantity_kwh
            sells[s].remaining_kwh -= quantity_kwh
        buy_allocation[b] -= quantity_kwh
        sell_allocation[s] -= quantity_kwh
        if buy_allocation[b] <= QUANTITY_EPSILON_KWH:
            b += 1
        if sell_allocation[s] <= QUANTITY_EPSILON_KWH:
            s += 1

    # The side whose whole level was allocated is done; anything its orders still
    # show open is float dust from the pairing, and must not look tradeable.
    for level, open_kwh in ((buys, open_buy_kwh), (sells, open_sell_kwh)):
        if volume_kwh >= open_kwh:
            for entry in level:
                entry.remaining_kwh = 0.0
    return matches


def _match_book(buys: list[_Entry], sells: list[_Entry]) -> list[_Match]:
    """Every match the book produces, best-priced levels first."""
    buy_levels = _price_levels(buys, highest_first=True)
    sell_levels = _price_levels(sells, highest_first=False)

    matches: list[_Match] = []
    b = s = 0
    while b < len(buy_levels) and s < len(sell_levels):
        buy_level = [entry for entry in buy_levels[b] if not entry.filled]
        sell_level = [entry for entry in sell_levels[s] if not entry.filled]
        if not buy_level:
            b += 1
            continue
        if not sell_level:
            s += 1
            continue

        # Best buy level no longer meets best sell level: no remaining pair can
        # cross either, because every later buy is cheaper and every later sell dearer.
        if (
            buy_level[0].order.limit_price_eur_per_kwh
            < sell_level[0].order.limit_price_eur_per_kwh
        ):
            break

        matches.extend(_match_levels(buy_level, sell_level))
    return matches


def clear_tick(
    tick: int,
    timestamp: datetime,
    orders: Sequence[AgentDecision],
) -> MarketState:
    """Clear one tick's order book and return the resulting market state.

    Orders are matched by price priority, with ties shared pro-rata (see module
    docstring). Unmatched orders — including the unfilled remainder of a partially
    filled one — are returned in ``MarketState`` so the caller can settle them
    against the household's grid import/export tariff instead. That fallback is what
    makes the P2P-versus-grid-only comparison meaningful.

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

    trades = [
        TradeEvent(
            trade_id=trade_id(tick, sequence),
            tick=tick,
            timestamp=timestamp,
            buyer_id=match.buy.order.household_id,
            seller_id=match.sell.order.household_id,
            quantity_kwh=match.quantity_kwh,
            clearing_price_eur_per_kwh=_clearing_price_eur_per_kwh(
                match.buy.order, match.sell.order
            ),
        )
        for sequence, match in enumerate(_match_book(buys, sells))
    ]

    return MarketState(
        tick=tick,
        timestamp=timestamp,
        trades=trades,
        unmatched_buy_orders=_unmatched(buys),
        unmatched_sell_orders=_unmatched(sells),
        clearing_price_eur_per_kwh=_volume_weighted_price(trades),
    )


def _unmatched(book: list[_Entry]) -> list[AgentDecision]:
    """Orders still open after clearing, in original arrival order.

    A partially filled order comes back with its *remaining* quantity, so the
    caller settles only the part the market could not fill. Fully filled orders
    are dropped rather than returned at quantity zero — ``AgentDecision``
    requires a positive quantity, and "nothing left to trade" is absence, not a
    zero-sized order.
    """
    still_open = sorted(
        (entry for entry in book if not entry.filled),
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
