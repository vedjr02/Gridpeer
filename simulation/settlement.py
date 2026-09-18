"""Grid-baseline fallback: what a tick actually cost each household, and what it
would have cost with no marketplace at all.

Build-order step 3/3 (see simulation/AGENTS.md). Orders the market could not match
do not vanish — the household still needs the energy, or still has the surplus. It
falls back to the grid at the tariffs on its own ``HouseholdProfile``: unmatched
buys are imported at the import tariff, unmatched sells are exported at the export
tariff.

That fallback is what makes the project's headline comparison possible. P2P trading
only "saves" anything relative to the grid-only counterfactual, so this module
reports both numbers for every household, every tick.

Orders are promises, meters are facts. An order is decided from a *forecast*, so what
a household trades rarely equals what it actually generated or consumed. Pass the
metered net position to ``settle_tick`` and the gap between the two — the imbalance —
is settled at grid tariffs: a seller that sold energy it did not have buys the
shortfall at the import tariff, a buyer that bought more than it used exports the
excess at the export tariff. Without this, a household could sell 50 kWh it never
generated and book the revenue as savings, which is the first thing an RL policy
would learn to do.

Sign convention throughout: **cost is positive when money leaves the household** and
negative when money comes in. A pure seller therefore has a negative cost, which is
revenue. Savings are ``grid_only - p2p``: positive means the marketplace left the
household better off.

This module deliberately stops at the per-tick figures. ``HouseholdOutcome`` and
``RunSummary`` are the dashboard's to build (see dashboard/AGENTS.md) — it sums
these across a run. Everything it needs is already here or on the shared contracts,
so no schema change is required to produce them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shared.schemas import AgentDecision, HouseholdProfile, MarketState, OrderSide


@dataclass(frozen=True)
class HouseholdSettlement:
    """One household's money for one tick, P2P versus grid-only. Internal to simulation/.

    All figures in EUR, positive = money out (see module docstring).
    """

    household_id: str
    tick: int
    traded_kwh: float
    """Energy actually traded on the marketplace this tick (bought or sold)."""

    grid_fallback_kwh: float
    """Energy settled at grid tariffs instead: unmatched orders, or, when settled
    against meters, any gap between what was traded and what was metered."""

    p2p_cost_eur: float
    """What the tick really cost: matched trades plus the grid fallback."""

    grid_only_cost_eur: float
    """Counterfactual: the same energy settled entirely at grid tariffs."""

    @property
    def savings_eur(self) -> float:
        """grid_only - p2p. Positive means the marketplace saved this household money."""
        return self.grid_only_cost_eur - self.p2p_cost_eur


def _grid_cost_eur(profile: HouseholdProfile, side: OrderSide, quantity_kwh: float) -> float:
    """Cost of settling ``quantity_kwh`` at this household's grid tariffs.

    Buying from the grid costs the import tariff; exporting to the grid earns the
    export tariff, hence the negative sign. The gap between those two tariffs is
    exactly the margin a P2P trade competes for.
    """
    if side is OrderSide.BUY:
        return quantity_kwh * profile.grid_import_tariff_eur_per_kwh
    return -quantity_kwh * profile.grid_export_tariff_eur_per_kwh


def _grid_cost_of_net_position_eur(profile: HouseholdProfile, net_position_kwh: float) -> float:
    """Cost of settling a physical net position entirely at this household's grid tariffs.

    Same sign convention as ``HouseholdState.net_position_kwh``: positive is surplus,
    exported at the export tariff; negative is deficit, imported at the import tariff.
    """
    if net_position_kwh >= 0:
        return _grid_cost_eur(profile, OrderSide.SELL, net_position_kwh)
    return _grid_cost_eur(profile, OrderSide.BUY, -net_position_kwh)


def settle_tick(
    state: MarketState,
    profiles: Mapping[str, HouseholdProfile],
    orders: Sequence[AgentDecision] | None = None,
    actual_net_position_kwh: Mapping[str, float] | None = None,
) -> dict[str, HouseholdSettlement]:
    """Settle one cleared tick for every household that took part in it.

    ``state`` is what ``clear_tick`` returned. ``profiles`` maps household_id to
    profile — tariffs differ per household, so a missing one is an error, not a
    default. ``orders`` is the original book; pass it when a household submitted an
    order that matched in full, so its grid-only counterfactual covers the whole
    order rather than only the unmatched part. Without it, the counterfactual is
    reconstructed from trades and unmatched orders, which is equivalent for every
    book ``clear_tick`` produces.

    ``actual_net_position_kwh`` maps household_id to what that household's meter
    recorded this tick (positive = surplus). Pass it whenever it is known: the tick
    is then settled against the meter rather than the orders (see
    :func:`_settle_against_meters`), and ``orders`` is not needed. Without it,
    settlement trusts the orders, which is only sound when every order exactly
    matches the household's real position — no forecast error, no over-trading.

    Returns one settlement per participating household, keyed by household_id.
    """
    if actual_net_position_kwh is not None:
        return _settle_against_meters(state, profiles, actual_net_position_kwh)

    totals: dict[str, dict[str, float]] = {}

    def bucket(household_id: str) -> dict[str, float]:
        if household_id not in profiles:
            raise KeyError(
                f"no HouseholdProfile for {household_id}; cannot settle a tick "
                f"without that household's grid tariffs"
            )
        return totals.setdefault(
            household_id,
            {"traded_kwh": 0.0, "grid_fallback_kwh": 0.0, "p2p": 0.0, "grid_only": 0.0},
        )

    # Matched trades: the buyer pays, the seller is paid, both at the clearing price.
    for trade in state.trades:
        value_eur = trade.quantity_kwh * trade.clearing_price_eur_per_kwh

        buyer = bucket(trade.buyer_id)
        buyer["traded_kwh"] += trade.quantity_kwh
        buyer["p2p"] += value_eur
        buyer["grid_only"] += _grid_cost_eur(
            profiles[trade.buyer_id], OrderSide.BUY, trade.quantity_kwh
        )

        seller = bucket(trade.seller_id)
        seller["traded_kwh"] += trade.quantity_kwh
        seller["p2p"] -= value_eur
        seller["grid_only"] += _grid_cost_eur(
            profiles[trade.seller_id], OrderSide.SELL, trade.quantity_kwh
        )

    # Unmatched orders: the household falls back to the grid, so P2P and grid-only
    # are identical for this portion — no saving, no loss. That is the honest
    # result, and it is why a market that matches nothing shows zero savings
    # rather than an artificial one.
    unmatched = list(state.unmatched_buy_orders) + list(state.unmatched_sell_orders)
    for order in unmatched:
        entry = bucket(order.household_id)
        cost_eur = _grid_cost_eur(profiles[order.household_id], order.side, order.quantity_kwh)
        entry["grid_fallback_kwh"] += order.quantity_kwh
        entry["p2p"] += cost_eur
        entry["grid_only"] += cost_eur

    # Households whose order matched in full appear in `trades` but not in the
    # unmatched books; those that never traded at all appear in neither. Walking
    # the original book catches the latter so a household that sat out a tick
    # still settles at zero rather than silently disappearing from the run.
    for order in orders or []:
        bucket(order.household_id)

    return {
        household_id: HouseholdSettlement(
            household_id=household_id,
            tick=state.tick,
            traded_kwh=entry["traded_kwh"],
            grid_fallback_kwh=entry["grid_fallback_kwh"],
            p2p_cost_eur=entry["p2p"],
            grid_only_cost_eur=entry["grid_only"],
        )
        for household_id, entry in totals.items()
    }


def _settle_against_meters(
    state: MarketState,
    profiles: Mapping[str, HouseholdProfile],
    actual_net_position_kwh: Mapping[str, float],
) -> dict[str, HouseholdSettlement]:
    """Settle a tick against metered net positions instead of trusting the orders.

    For each household:

        net_sold_kwh   = kWh sold on the market - kWh bought on the market
        imbalance_kwh  = actual_net_position_kwh - net_sold_kwh
        p2p cost       = market purchases - market sales + imbalance at grid tariffs
        grid-only cost = actual_net_position_kwh at grid tariffs

    Unmatched orders need no separate treatment: energy the market did not take is
    part of the imbalance, so when orders equal meters this gives exactly the same
    figures as order-based settlement. Every metered household is settled, including
    ones that did not trade — they still imported or exported, and saved nothing.
    """
    traded_kwh: dict[str, float] = {}
    net_sold_kwh: dict[str, float] = {}
    market_cost_eur: dict[str, float] = {}

    for trade in state.trades:
        value_eur = trade.quantity_kwh * trade.clearing_price_eur_per_kwh
        for household_id, direction in ((trade.buyer_id, 1.0), (trade.seller_id, -1.0)):
            traded_kwh[household_id] = traded_kwh.get(household_id, 0.0) + trade.quantity_kwh
            net_sold_kwh[household_id] = (
                net_sold_kwh.get(household_id, 0.0) - direction * trade.quantity_kwh
            )
            market_cost_eur[household_id] = (
                market_cost_eur.get(household_id, 0.0) + direction * value_eur
            )

    unmetered = sorted(set(traded_kwh) - set(actual_net_position_kwh))
    if unmetered:
        raise KeyError(
            f"no metered net position for {', '.join(unmetered)}; a household that "
            f"traded cannot be settled without knowing what it actually delivered"
        )

    settlements: dict[str, HouseholdSettlement] = {}
    for household_id, actual_kwh in actual_net_position_kwh.items():
        if household_id not in profiles:
            raise KeyError(
                f"no HouseholdProfile for {household_id}; cannot settle a tick "
                f"without that household's grid tariffs"
            )
        profile = profiles[household_id]
        imbalance_kwh = actual_kwh - net_sold_kwh.get(household_id, 0.0)
        settlements[household_id] = HouseholdSettlement(
            household_id=household_id,
            tick=state.tick,
            traded_kwh=traded_kwh.get(household_id, 0.0),
            grid_fallback_kwh=abs(imbalance_kwh),
            p2p_cost_eur=(
                market_cost_eur.get(household_id, 0.0)
                + _grid_cost_of_net_position_eur(profile, imbalance_kwh)
            ),
            grid_only_cost_eur=_grid_cost_of_net_position_eur(profile, actual_kwh),
        )
    return settlements
