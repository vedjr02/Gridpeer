"""Evaluation harness: run a strategy end to end and report what it achieved.

agents/CLAUDE.md calls this the module's most important deliverable — more than
the RL training itself — and says to build it early rather than improvise it under
deadline pressure. It is the thing that turns a simulation into the project's
single defensible sentence: *these agents saved households X% versus exporting to
the grid.*

Running a strategy means, per tick: ask the forecaster what each household expects,
ask the strategy what to do about it, clear the resulting book, and settle the
result against each household's grid tariffs. Swapping the baseline for a trained
policy changes exactly one argument, which is the point — the comparison is only
meaningful if both strategies run through identical machinery.

Two things this harness does NOT yet do, both of which need a team decision
rather than a unilateral one, and both flagged in the module README:

1. **Batteries take no part.** Households trade their forecast net position, which
   is solar minus demand with no storage in between. simulation/ has a working
   battery model (HouseholdEnvironment), but wiring it in first requires agreeing
   whether the battery is environment-owned physics or an agent decision variable —
   the forecaster predicts raw demand and solar, so today nothing owns that choice.
   Until it is settled, storage sits idle and the savings reported here are the
   no-storage floor.
2. **RunSummary is also listed as a dashboard output** in dashboard/CLAUDE.md. The
   rollup is computed here because a sibling import would break the module
   boundary, but the arithmetic should live in one place before two versions drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from forecasting.baseline import NaiveForecaster
from shared.schemas import (
    AgentDecision,
    ForecastOutput,
    HouseholdOutcome,
    HouseholdProfile,
    MarketState,
    OrderSide,
    RunSummary,
)
from simulation.market import clear_tick
from simulation.settlement import settle_tick

TICK_MINUTES = 30
_EPOCH = datetime(2026, 1, 1, 0, 0)

# Ireland's grid carbon intensity, kg CO2 per kWh. A round working figure, not a
# citation — the final writeup should source this properly (SEAI publishes it
# annually) because it scales the headline CO2 number directly.
DEFAULT_GRID_CO2_KG_PER_KWH = 0.30


class TradingStrategy(Protocol):
    """What the harness needs from any strategy, baseline or learned.

    Deliberately the narrowest possible interface: an RL policy satisfies it by
    exposing the same ``decide`` method, so nothing here has to change when the
    trained policies arrive.
    """

    def decide(
        self, forecast: ForecastOutput, profile: HouseholdProfile
    ) -> AgentDecision | None:
        """Return this household's order for the tick, or None to sit it out."""
        ...


@dataclass(frozen=True)
class HouseholdSeries:
    """A household's profile plus its observed demand and solar series, in kWh."""

    profile: HouseholdProfile
    demand_kwh: list[float]
    solar_kwh: list[float]


@dataclass(frozen=True)
class RunResult:
    """Everything one run produced: the summary, the per-household detail, the ticks.

    The summary is the number that gets quoted; the rest is what makes it
    defensible when somebody asks how it was arrived at.
    """

    summary: RunSummary
    outcomes: list[HouseholdOutcome]
    market_states: list[MarketState]


def _timestamp(tick: int) -> datetime:
    """Wall-clock time for a tick, at the half-hourly resolution of the CER data."""
    return _EPOCH + timedelta(minutes=TICK_MINUTES * tick)


def _peak_grid_import_kwh(per_tick_imports_kwh: Sequence[float]) -> float:
    """Highest simultaneous grid draw across the run — the number that strains a grid."""
    return max(per_tick_imports_kwh, default=0.0)


def run_strategy(
    households: Sequence[HouseholdSeries],
    strategy: TradingStrategy,
    run_id: str,
    strategy_name: str,
    num_ticks: int | None = None,
    forecast_window: int = 4,
    grid_co2_kg_per_kwh: float = DEFAULT_GRID_CO2_KG_PER_KWH,
) -> RunResult:
    """Run one strategy over the households' series and summarise the outcome.

    Forecasts are built from history up to the current tick only — never from the
    tick being decided — so the strategy is judged on what it could actually have
    known at the time.
    """
    if not households:
        raise ValueError("no households to run")

    available = min(len(h.demand_kwh) for h in households)
    ticks = available if num_ticks is None else min(num_ticks, available)
    profiles = {h.profile.household_id: h.profile for h in households}
    forecaster = NaiveForecaster(window=forecast_window)

    market_states: list[MarketState] = []
    p2p_cost_eur: dict[str, float] = {hid: 0.0 for hid in profiles}
    grid_cost_eur: dict[str, float] = {hid: 0.0 for hid in profiles}
    traded_kwh_total = 0.0
    p2p_imports_per_tick: list[float] = []
    grid_imports_per_tick: list[float] = []

    for tick in range(ticks):
        timestamp = _timestamp(tick)

        orders: list[AgentDecision] = []
        for household in households:
            forecast = forecaster.forecast(
                household_id=household.profile.household_id,
                tick=tick,
                timestamp=timestamp,
                demand_history_kwh=household.demand_kwh[:tick],
                solar_history_kwh=household.solar_kwh[:tick],
            )
            decision = strategy.decide(forecast, household.profile)
            if decision is not None:
                orders.append(decision)

        state = clear_tick(tick=tick, timestamp=timestamp, orders=orders)
        market_states.append(state)

        for household_id, settlement in settle_tick(state, profiles, orders).items():
            p2p_cost_eur[household_id] += settlement.p2p_cost_eur
            grid_cost_eur[household_id] += settlement.grid_only_cost_eur
        traded_kwh_total += sum(trade.quantity_kwh for trade in state.trades)

        # Peak load: under P2P only the unmatched buys are drawn from the grid,
        # whereas with no marketplace every buy order would be.
        p2p_imports_per_tick.append(
            sum(order.quantity_kwh for order in state.unmatched_buy_orders)
        )
        grid_imports_per_tick.append(
            sum(order.quantity_kwh for order in orders if order.side is OrderSide.BUY)
        )

    outcomes = [_outcome(hid, p2p_cost_eur[hid], grid_cost_eur[hid]) for hid in profiles]
    peak_p2p = _peak_grid_import_kwh(p2p_imports_per_tick)
    peak_grid = _peak_grid_import_kwh(grid_imports_per_tick)

    summary = RunSummary(
        run_id=run_id,
        strategy_name=strategy_name,
        num_households=len(households),
        num_ticks=ticks,
        total_savings_eur=sum(o.savings_eur for o in outcomes),
        avg_savings_pct=_avg_savings_pct(outcomes),
        peak_load_reduction_pct=(
            (peak_grid - peak_p2p) / peak_grid * 100.0 if peak_grid > 0 else 0.0
        ),
        # Energy sourced from a neighbour's panels is energy not drawn from the
        # grid. Stated as an assumption, because it is one.
        co2_avoided_kg=traded_kwh_total * grid_co2_kg_per_kwh,
    )
    return RunResult(summary=summary, outcomes=outcomes, market_states=market_states)


def _avg_savings_pct(outcomes: Sequence[HouseholdOutcome]) -> float:
    """Savings across the run as a share of what the grid-only baseline would have cost.

    Weighted by each household's baseline, not a plain mean of the per-household
    percentages. A net seller's baseline is small (it only ever earned the export
    tariff), so its percentage is huge and a plain mean is dominated by it — one
    small producer could report a 100%+ "average saving" for the whole run. The
    weighted figure is the one that survives being questioned.
    """
    reference_eur = sum(abs(outcome.total_cost_eur_grid_baseline) for outcome in outcomes)
    if reference_eur <= 0:
        return 0.0
    return sum(outcome.savings_eur for outcome in outcomes) / reference_eur * 100.0


def _outcome(
    household_id: str, p2p_cost_eur: float, grid_cost_eur: float
) -> HouseholdOutcome:
    """Roll one household's run up into the dashboard's per-household contract.

    ``savings_pct`` is measured against the magnitude of the grid-only figure, so a
    net seller (whose costs are negative revenue) reports a positive percentage for
    earning more, rather than a sign-flipped one.
    """
    savings_eur = grid_cost_eur - p2p_cost_eur
    reference_eur = abs(grid_cost_eur)
    return HouseholdOutcome(
        household_id=household_id,
        total_cost_eur_p2p=p2p_cost_eur,
        total_cost_eur_grid_baseline=grid_cost_eur,
        savings_eur=savings_eur,
        savings_pct=(savings_eur / reference_eur * 100.0) if reference_eur > 0 else 0.0,
    )


def compare(baseline: RunSummary, candidate: RunSummary) -> str:
    """One line of plain English comparing two runs — the demo's closing sentence.

    Reports the honest direction either way: agents/CLAUDE.md is explicit that a
    learned policy failing to beat the baseline is a legitimate result, and a
    harness that could only phrase a win would quietly encourage hiding a loss.
    """
    delta_eur = candidate.total_savings_eur - baseline.total_savings_eur
    verb = "more" if delta_eur >= 0 else "less"
    return (
        f"{candidate.strategy_name}: EUR {candidate.total_savings_eur:.2f} saved "
        f"({candidate.avg_savings_pct:.1f}% avg) vs {baseline.strategy_name}: "
        f"EUR {baseline.total_savings_eur:.2f} ({baseline.avg_savings_pct:.1f}% avg) "
        f"— EUR {abs(delta_eur):.2f} {verb}, over {candidate.num_ticks} ticks "
        f"and {candidate.num_households} households."
    )
