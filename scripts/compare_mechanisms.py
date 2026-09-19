"""Does the market's pricing rule change what households save?

    python scripts/compare_mechanisms.py

Runs identical households through the same forecasts and the same rule-based
baseline twice — once with each ``PricingRule`` — and reports what changed. Because
matching is identical under both rules (see simulation/market.py), the energy traded
is the same in both runs, so any difference in savings is caused by the price rule
alone, not by who managed to trade.

What it reports per rule:

- **community savings**: the marketplace's total saving against grid-only settlement;
- **buyers' / sellers' share**: how that saving is split between the two sides;
- **price spread**: the average gap between the dearest and cheapest trade within a
  tick — how differently two neighbours can be charged for the same half-hour.

What to expect, and why: **community savings come out identical under both rules.**
What one neighbour pays for a kWh, another receives, so with the same trades the
price can only move savings between households, never create them. The rule changes
*who* saves and whether neighbours pay the same in a half-hour. Where it could change
the total is through behaviour — a strategy that learns under one rule may bid
differently under the other — which is the experiment to run with the RL policies.

Like scripts/compare_strategies.py, this composes modules from outside them: the
market lives in simulation/, the strategy in agents/, the forecaster in forecasting/,
and the households in data/.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.baseline import RuleBasedTrader
from data.synthetic import default_profiles, generate_household_series
from forecasting.baseline import NaiveForecaster
from simulation.market import PricingRule
from simulation.simulator import HouseholdSeries, MarketSimulator, timestamp_for

COMMUNITY_SIZES = (4, 20, 100)
DAYS = 7
SEED = 0


@dataclass
class MechanismOutcome:
    """What one pricing rule produced over a run, in EUR and kWh."""

    savings_eur: float = 0.0
    buyer_savings_eur: float = 0.0
    seller_savings_eur: float = 0.0
    traded_kwh: float = 0.0
    price_spread_sum_eur_per_kwh: float = 0.0
    ticks_with_trades: int = 0

    @property
    def mean_price_spread_eur_per_kwh(self) -> float:
        """Average (dearest - cheapest) trade price within a tick that traded."""
        if not self.ticks_with_trades:
            return 0.0
        return self.price_spread_sum_eur_per_kwh / self.ticks_with_trades


def community(num_households: int, days: int = DAYS, seed: int = SEED) -> list[HouseholdSeries]:
    """The synthetic cohort from data/, as the simulator's household series."""
    households = []
    for profile in default_profiles(num_households):
        demand_kwh, solar_kwh = generate_household_series(profile, days=days, seed=seed)
        households.append(HouseholdSeries(profile, demand_kwh, solar_kwh))
    return households


def run(households: list[HouseholdSeries], pricing: PricingRule) -> MechanismOutcome:
    """Trade the baseline over the households under one pricing rule."""
    simulator = MarketSimulator(households, pricing=pricing)
    forecaster = NaiveForecaster(window=4)
    trader = RuleBasedTrader()
    outcome = MechanismOutcome()

    for tick in range(simulator.num_ticks):
        timestamp = timestamp_for(tick)
        orders = []
        for household in households:
            household_id = household.profile.household_id
            forecast = forecaster.forecast(
                household_id=household_id,
                tick=tick,
                timestamp=timestamp,
                demand_history_kwh=household.demand_kwh[:tick],
                solar_history_kwh=household.solar_kwh[:tick],
            )
            # The baseline sees its battery, as it does in the dashboard pipeline.
            state = simulator.environments[household_id].contract_state(timestamp)
            order = trader.decide(forecast, household.profile, state)
            if order is not None:
                orders.append(order)

        result = simulator.step(orders)
        trades = result.market_state.trades
        buyers = {trade.buyer_id for trade in trades}
        sellers = {trade.seller_id for trade in trades}

        for household_id, settlement in result.settlements.items():
            outcome.savings_eur += settlement.savings_eur
            if household_id in buyers:
                outcome.buyer_savings_eur += settlement.savings_eur
            elif household_id in sellers:
                outcome.seller_savings_eur += settlement.savings_eur

        if trades:
            prices = [trade.clearing_price_eur_per_kwh for trade in trades]
            outcome.traded_kwh += sum(trade.quantity_kwh for trade in trades)
            outcome.price_spread_sum_eur_per_kwh += max(prices) - min(prices)
            outcome.ticks_with_trades += 1
    return outcome


def compare(
    num_households: int, days: int = DAYS, seed: int = SEED
) -> dict[PricingRule, MechanismOutcome]:
    """Both pricing rules on one identical community."""
    households = community(num_households, days, seed)
    return {pricing: run(households, pricing) for pricing in PricingRule}


def _share(part: float, whole: float) -> str:
    return f"{part / whole * 100:5.1f}%" if whole else "   n/a"


def main() -> None:
    """Print the comparison for each community size."""
    print(f"Rule-based baseline, {DAYS} days, synthetic households (seed {SEED})\n")
    header = (
        f"{'households':>10}  {'pricing rule':<18} {'savings':>10} {'buyers':>7} "
        f"{'sellers':>7} {'traded':>9} {'price spread':>13}"
    )
    print(header)
    print("-" * len(header))
    for size in COMMUNITY_SIZES:
        outcomes = compare(size)
        for pricing, outcome in outcomes.items():
            print(
                f"{size:>10}  {pricing.value:<18} EUR {outcome.savings_eur:6.2f} "
                f"{_share(outcome.buyer_savings_eur, outcome.savings_eur)} "
                f"{_share(outcome.seller_savings_eur, outcome.savings_eur)} "
                f"{outcome.traded_kwh:6.1f} kWh "
                f"{outcome.mean_price_spread_eur_per_kwh * 100:7.2f} c/kWh"
            )
        print()


if __name__ == "__main__":
    main()
