"""Run the rule-based baseline over a small demo set and print the result.

    python -m agents.run_baseline

A smoke test you can read: it exercises forecasting -> agents -> simulation ->
settlement in one command and prints the numbers the dashboard will eventually
show. The households here are placeholders standing in for the data/ loader —
a steady producer, a steady consumer, and a prosumer whose surplus and demand
move over the day.
"""

from __future__ import annotations

import math

from agents.baseline import STRATEGY_NAME, RuleBasedTrader
from agents.evaluation import TICK_MINUTES, HouseholdSeries, run_strategy
from shared.schemas import AgentRole, HouseholdProfile

TICKS_PER_DAY = 24 * 60 // TICK_MINUTES  # 48 half-hours


def _profile(
    household_id: str, role: AgentRole, battery_capacity_kwh: float
) -> HouseholdProfile:
    """A demo household on typical Irish residential tariffs."""
    has_solar = role is not AgentRole.PURE_CONSUMER
    return HouseholdProfile(
        household_id=household_id,
        role=role,
        has_solar=has_solar,
        solar_capacity_kw=3.0 if has_solar else 0.0,
        battery_capacity_kwh=battery_capacity_kwh,
        grid_export_tariff_eur_per_kwh=0.07,
        grid_import_tariff_eur_per_kwh=0.25,
    )


def _daily_shape(tick: int, role: AgentRole, solar_capacity_kw: float) -> tuple[float, float]:
    """Placeholder half-hourly demand and solar for one tick (shape, not truth).

    Morning and evening demand peaks, a midday solar bell — the shape of a real
    residential load curve without pretending to be one. Replaced by the data/
    loader once real CER/PVGIS data lands.
    """
    hour = (tick % TICKS_PER_DAY) * TICK_MINUTES / 60.0
    morning = math.exp(-((hour - 8) ** 2) / 2.0)
    evening = math.exp(-((hour - 19) ** 2) / 2.0)
    base = 0.15 if role is AgentRole.PURE_PRODUCER else 0.3
    peaks = 0.0 if role is AgentRole.PURE_PRODUCER else 1.2 * (morning + evening)
    demand = base + peaks

    solar = 0.0
    if solar_capacity_kw > 0 and 6 <= hour <= 20:
        solar = solar_capacity_kw * 0.5 * math.exp(-((hour - 13) ** 2) / 8.0)
    return demand, solar


def demo_households(days: int = 2) -> list[HouseholdSeries]:
    """Three households that can plausibly trade with each other."""
    specs = [
        ("hh_producer", AgentRole.PURE_PRODUCER, 0.0),
        ("hh_consumer", AgentRole.PURE_CONSUMER, 0.0),
        ("hh_prosumer", AgentRole.PROSUMER, 5.0),
    ]
    households: list[HouseholdSeries] = []
    for household_id, role, battery_kwh in specs:
        profile = _profile(household_id, role, battery_kwh)
        series = [
            _daily_shape(tick, role, profile.solar_capacity_kw)
            for tick in range(days * TICKS_PER_DAY)
        ]
        households.append(
            HouseholdSeries(
                profile=profile,
                demand_kwh=[demand for demand, _ in series],
                solar_kwh=[solar for _, solar in series],
            )
        )
    return households


def main() -> None:
    """Run the baseline over the demo households and print the outcome."""
    households = demo_households()
    result = run_strategy(
        households,
        RuleBasedTrader(),
        run_id="baseline_demo",
        strategy_name=STRATEGY_NAME,
    )
    summary = result.summary

    print(
        f"{summary.strategy_name}: {summary.num_households} households "
        f"x {summary.num_ticks} ticks"
    )
    print(f"  total savings      EUR {summary.total_savings_eur:7.2f}")
    print(f"  average savings        {summary.avg_savings_pct:7.1f}%  (weighted by baseline cost)")
    print(f"  peak load reduction    {summary.peak_load_reduction_pct:7.1f}%")
    print(f"  CO2 avoided            {summary.co2_avoided_kg:7.2f} kg  (at 0.30 kg/kWh)")
    print()
    print("  per household (cost positive = money out):")
    for outcome in result.outcomes:
        print(
            f"    {outcome.household_id:<12} p2p {outcome.total_cost_eur_p2p:+7.2f}  "
            f"grid-only {outcome.total_cost_eur_grid_baseline:+7.2f}  "
            f"saved {outcome.savings_eur:+6.2f} ({outcome.savings_pct:5.1f}%)"
        )

    traded = sum(len(state.trades) for state in result.market_states)
    quiet = sum(1 for state in result.market_states if not state.trades)
    print()
    print(f"  {traded} trades cleared; {quiet} ticks cleared nothing (overnight, or cold start)")


if __name__ == "__main__":
    main()
