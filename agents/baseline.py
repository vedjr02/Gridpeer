"""Rule-based baseline trader: the yardstick every RL result is measured against.

Build-order step 1 (see agents/CLAUDE.md), and explicitly not a throwaway. Every
claim this project makes has the shape "the learned policy beat *this* by X%", so
the baseline has to be reasonable enough that beating it means something, and
simple enough that nobody has to wonder why it did what it did.

The rule, in full:

1. **Side and quantity** come from the forecast net position. Positive means
   surplus to sell, negative means a deficit to buy; the quantity is its size.
2. **Price** is interpolated inside the household's own tariff gap — between what
   the grid pays for an export and what the grid charges for an import. That gap
   is the only margin a P2P trade can capture, so a baseline that prices outside
   it would be strictly worse than not trading at all.
3. **Eagerness** slides the price within that gap, and rises with how much energy
   the household has at stake: a big surplus is worth giving up margin for,
   because the alternative is dumping it to the grid at the export tariff.

With both sides on this rule and an eagerness of at least 0.5, asks always land at
or below bids, so the book crosses. That is deliberate — a baseline that rarely
trades would flatter any RL policy compared against it.
"""

from __future__ import annotations

from shared.schemas import AgentDecision, ForecastOutput, HouseholdProfile, OrderSide

STRATEGY_NAME = "rule_based_baseline"

# Below this, a position is not worth an order (1 Wh). AgentDecision requires a
# positive quantity, so "nothing to trade" is no order at all, not a zero one.
MIN_TRADEABLE_KWH = 1e-3


class RuleBasedTrader:
    """Turns one household's forecast into one order, by the rule above.

    Stateless: every tick is decided from that tick's forecast and the household's
    own tariffs. Nothing carries over, which makes any single decision reproducible
    in isolation — the property that makes this useful as a debugging reference
    once the RL policies start behaving strangely.
    """

    def __init__(
        self,
        base_eagerness: float = 0.5,
        urgency_gain: float = 0.3,
        urgency_scale_kwh: float = 2.0,
    ) -> None:
        """base_eagerness: share of the tariff gap conceded at zero urgency.

        0.5 prices at the midpoint of the gap — an even split of the surplus with
        the counterparty. Below 0.5 the book stops crossing; above it, the
        household concedes more to trade more reliably.

        urgency_gain: how much eagerness a fully urgent position adds.
        urgency_scale_kwh: the position size treated as fully urgent.
        """
        if not 0.5 <= base_eagerness <= 1.0:
            raise ValueError(
                f"base_eagerness must be in [0.5, 1.0] — below 0.5 asks sit above bids "
                f"and the baseline stops trading entirely (got {base_eagerness})"
            )
        if urgency_gain < 0:
            raise ValueError("urgency_gain must be >= 0")
        if urgency_scale_kwh <= 0:
            raise ValueError("urgency_scale_kwh must be > 0")
        self.base_eagerness = base_eagerness
        self.urgency_gain = urgency_gain
        self.urgency_scale_kwh = urgency_scale_kwh

    def _eagerness(self, quantity_kwh: float) -> float:
        """How much of the tariff gap to concede, in [0, 1], rising with size."""
        urgency = min(1.0, quantity_kwh / self.urgency_scale_kwh)
        return min(1.0, self.base_eagerness + self.urgency_gain * urgency)

    def _limit_price_eur_per_kwh(
        self, profile: HouseholdProfile, side: OrderSide, quantity_kwh: float
    ) -> float:
        """Price this order inside the household's import/export tariff gap.

        A seller's floor is the export tariff (the grid alternative) and a buyer's
        ceiling is the import tariff. Eagerness moves the price from the greedy end
        of that range toward the grid-equivalent end.
        """
        floor = profile.grid_export_tariff_eur_per_kwh
        ceiling = profile.grid_import_tariff_eur_per_kwh
        gap = max(0.0, ceiling - floor)
        eagerness = self._eagerness(quantity_kwh)

        if side is OrderSide.SELL:
            # Greedy ask is the import tariff; conceding walks it down to the export tariff.
            return floor + (1.0 - eagerness) * gap
        # Greedy bid is the export tariff; conceding walks it up to the import tariff.
        return ceiling - (1.0 - eagerness) * gap

    def decide(
        self, forecast: ForecastOutput, profile: HouseholdProfile
    ) -> AgentDecision | None:
        """One household's order for one tick, or None if it has nothing to trade.

        ``forecast`` and ``profile`` must describe the same household. Returns None
        when the forecast net position is effectively zero — a household in balance
        should sit the tick out rather than submit a token order.
        """
        if forecast.household_id != profile.household_id:
            raise ValueError(
                f"forecast is for {forecast.household_id} but profile is for "
                f"{profile.household_id}"
            )

        net_position_kwh = forecast.predicted_net_position_kwh
        quantity_kwh = abs(net_position_kwh)
        if quantity_kwh < MIN_TRADEABLE_KWH:
            return None

        side = OrderSide.SELL if net_position_kwh > 0 else OrderSide.BUY
        return AgentDecision(
            household_id=forecast.household_id,
            tick=forecast.tick,
            side=side,
            quantity_kwh=quantity_kwh,
            limit_price_eur_per_kwh=self._limit_price_eur_per_kwh(
                profile, side, quantity_kwh
            ),
            strategy_name=STRATEGY_NAME,
        )
