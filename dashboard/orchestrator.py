"""Orchestration loop: ticks the whole pipeline forward, one stage at a time.

The loop, per tick:

    forecasting  (ForecastOutput)   -> agents      (AgentDecision)
                                    -> simulation  (MarketState/TradeEvent)
                                    -> persistence (RunStore)
                                    -> outcomes    (HouseholdOutcome/RunSummary)

Every arrow above is a shared contract from ``shared/schemas.py``. The loop calls
``agents`` and ``simulation`` through their public entry points and never reaches
into their internals — it owns the sequencing and nothing else. That is the whole
job: keep it boring, keep it tested, and let the four modules stay swappable.

Two deliberate positions, both worth knowing before quoting any number from a run:

1. **Settlement is against meters, not orders.** Orders come from a *forecast*, so
   what a household offered rarely equals what it actually generated or consumed.
   ``settle_tick`` is given the metered net position, and the gap is settled at grid
   tariffs. Without that, a household could sell energy it never had and book the
   revenue as savings.
2. **Batteries are off by default** (``use_battery=False``). simulation/ has a
   working battery model, but whether storage is environment-owned physics or an
   agent decision variable is an open team question (flagged in agents/README.md
   and agents/evaluation.py). Until it is settled, the default matches the rest of
   the codebase and the reported savings are the no-storage floor.

Household series come from ``data.synthetic`` when that module exists, and from the
local placeholder generator below when it does not — see :func:`_synthetic_series`.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from agents.baseline import STRATEGY_NAME, RuleBasedTrader
from dashboard.outcomes import (
    IE_GRID_CO2_G_PER_KWH,
    compute_household_outcomes,
    compute_run_summary,
)
from dashboard.persistence import RunStore
from forecasting.baseline import NaiveForecaster
from shared.schemas import (
    AgentDecision,
    AgentRole,
    ForecastOutput,
    HouseholdOutcome,
    HouseholdProfile,
    MarketState,
    RunSummary,
    TradeEvent,
)
from simulation.environment import HouseholdEnvironment
from simulation.market import clear_tick
from simulation.settlement import settle_tick

TICK_MINUTES = 30

#: Where ``main`` persists a run. The dashboard reads the same default, so
#: ``make run-sim`` followed by ``streamlit run dashboard/app.py`` shows that run.
#: Override with GRIDPEER_DB on both sides to point at another database.
DEFAULT_DB_PATH = Path(os.environ.get("GRIDPEER_DB", "data/gridpeer.db"))
TICKS_PER_DAY = 24 * 60 // TICK_MINUTES  # 48, matching the CER data's resolution
_EPOCH = datetime(2026, 1, 1, 0, 0)


def _timestamp(tick: int) -> datetime:
    """Wall-clock time for a tick, at half-hour (CER) resolution."""
    return _EPOCH + timedelta(minutes=TICK_MINUTES * tick)


class TradingStrategy(Protocol):
    """What the loop needs from a strategy: one order per household per tick.

    Structural on purpose — the rule-based baseline satisfies it today and a
    trained policy satisfies it by exposing the same ``decide``, so nothing in
    this loop changes when the RL agents land.
    """

    def decide(
        self, forecast: ForecastOutput, profile: HouseholdProfile
    ) -> AgentDecision | None:
        """Return this household's order for the tick, or None to sit it out."""
        ...


@dataclass(frozen=True)
class HouseholdSeries:
    """A household's identity plus its observed demand/solar series (kWh).

    The series are the ground truth: the forecaster sees them only up to the current
    tick, while settlement sees the current tick's actual value. That split is what
    makes forecast error show up in the savings figure instead of being assumed away.
    """

    profile: HouseholdProfile
    demand_kwh: list[float]
    solar_kwh: list[float]


@dataclass(frozen=True)
class PipelineResult:
    """Everything one run produced, in the order the dashboard wants to render it."""

    run_id: str
    strategy_name: str
    forecasts: list[ForecastOutput] = field(default_factory=list)
    decisions: list[AgentDecision] = field(default_factory=list)
    market_states: list[MarketState] = field(default_factory=list)
    outcomes: list[HouseholdOutcome] = field(default_factory=list)
    summary: RunSummary | None = None

    @property
    def trades(self) -> list[TradeEvent]:
        """Every trade cleared in the run, in tick order."""
        return [trade for state in self.market_states for trade in state.trades]


# ---------------------------------------------------------------------------
# Household series: data/ when it exists, placeholder generator when it does not
# ---------------------------------------------------------------------------

def _placeholder_series(
    role: AgentRole, solar_capacity_kw: float, days: int, seed: int
) -> tuple[list[float], list[float]]:
    """Placeholder half-hourly demand + solar for one household (shape, not truth).

    Stand-in for ``data.synthetic`` while that module is being built: morning and
    evening demand peaks, a daylight solar bell curve. Right shape, invented values.
    """
    demand: list[float] = []
    solar: list[float] = []
    for tick in range(days * TICKS_PER_DAY):
        hour = (tick % TICKS_PER_DAY) * TICK_MINUTES / 60.0
        jitter = ((tick * 2654435761 + seed) % 1000) / 1000.0 * 0.2
        morning = math.exp(-((hour - 8) ** 2) / 2.0)
        evening = math.exp(-((hour - 19) ** 2) / 2.0)
        base = 0.15 if role == AgentRole.PURE_PRODUCER else 0.3
        peaks = 0.0 if role == AgentRole.PURE_PRODUCER else 1.2 * (morning + evening)
        demand.append(base + peaks + jitter)
        if solar_capacity_kw > 0 and 6 <= hour <= 20:
            midday = math.exp(-((hour - 13) ** 2) / 8.0)
            solar.append(max(0.0, solar_capacity_kw * 0.5 * midday))
        else:
            solar.append(0.0)
    return demand, solar


def _synthetic_series(
    profile: HouseholdProfile, days: int, seed: int
) -> tuple[list[float], list[float]]:
    """One household's (demand_kwh, solar_kwh) series from data/, or the placeholder.

    ``data/synthetic.py`` is owned by another task. This is the single point the
    pipeline couples to it, coded against its published signature:

        generate_household_series(
            profile: HouseholdProfile,
            days: int,
            tick_minutes: int,
            seed: int,
        ) -> tuple[Sequence[float], Sequence[float]]   # (demand_kwh, solar_kwh)

    A missing module falls back to :func:`_placeholder_series`, so nobody is blocked
    on it. A module that is present but whose signature has drifted raises instead of
    falling back — silently reverting to invented data would be worse than failing,
    because the run would still produce a number and nobody would know it was fake.
    """
    try:
        from data.synthetic import generate_household_series  # type: ignore[import-not-found]
    except (ImportError, AttributeError):
        return _placeholder_series(
            profile.role, profile.solar_capacity_kw, days=days, seed=seed
        )

    try:
        demand_kwh, solar_kwh = generate_household_series(
            profile=profile,
            days=days,
            tick_minutes=TICK_MINUTES,
            seed=seed,
        )
    except TypeError as error:  # signature drifted away from the agreed contract
        raise TypeError(
            "data.synthetic.generate_household_series no longer matches the contract "
            "dashboard/orchestrator.py codes against (see _synthetic_series). Update "
            "this one function rather than working around it elsewhere."
        ) from error
    return list(demand_kwh), list(solar_kwh)


def demo_households(num_households: int = 4, days: int = 2) -> list[HouseholdSeries]:
    """Build a small mixed set of households with their demand/solar series.

    A deliberately mixed book — consumers, prosumers and a pure producer — because a
    market of identical households has nobody to trade with and reports zero savings.
    """
    roles = [
        AgentRole.PROSUMER,
        AgentRole.PURE_CONSUMER,
        AgentRole.PROSUMER,
        AgentRole.PURE_PRODUCER,
    ]
    out: list[HouseholdSeries] = []
    for i in range(num_households):
        role = roles[i % len(roles)]
        has_solar = role in (AgentRole.PROSUMER, AgentRole.PURE_PRODUCER)
        solar_capacity = 3.0 if has_solar else 0.0
        profile = HouseholdProfile(
            household_id=f"hh_{i:03d}",
            role=role,
            has_solar=has_solar,
            solar_capacity_kw=solar_capacity,
            battery_capacity_kwh=5.0 if has_solar else 0.0,
            grid_export_tariff_eur_per_kwh=0.07,
            grid_import_tariff_eur_per_kwh=0.25,
        )
        demand, solar = _synthetic_series(profile, days=days, seed=i)
        out.append(HouseholdSeries(profile=profile, demand_kwh=demand, solar_kwh=solar))
    return out


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def _tick_count(households: Sequence[HouseholdSeries], num_ticks: int | None) -> int:
    """Ticks this run can cover: the shortest series, capped by ``num_ticks``."""
    if not households:
        raise ValueError("no households to run")
    available = min(len(h.demand_kwh) for h in households)
    return available if num_ticks is None else min(num_ticks, available)


def run_forecasts(
    households: list[HouseholdSeries],
    num_ticks: int | None = None,
    window: int = 4,
) -> list[ForecastOutput]:
    """Tick the forecasting stage alone and collect ForecastOutput.

    Kept separate from :func:`run_pipeline` because the forecast view in the
    dashboard does not need a market. Each household is forecast from its own
    history up to that tick only — never from the tick being predicted.
    """
    ticks = _tick_count(households, num_ticks)
    forecaster = NaiveForecaster(window=window)
    return [
        forecaster.forecast(
            household_id=household.profile.household_id,
            tick=tick,
            timestamp=_timestamp(tick),
            demand_history_kwh=household.demand_kwh[:tick],
            solar_history_kwh=household.solar_kwh[:tick],
        )
        for tick in range(ticks)
        for household in households
    ]


def _metered_net_positions(
    households: Sequence[HouseholdSeries], ticks: int, use_battery: bool
) -> list[dict[str, float]]:
    """Per-tick metered net position (kWh, + surplus) for every household.

    With ``use_battery`` the household's battery absorbs what it can first, via
    simulation's ``HouseholdEnvironment``; without it the net position is simply
    solar minus demand.
    """
    if not use_battery:
        return [
            {
                household.profile.household_id: (
                    household.solar_kwh[tick] - household.demand_kwh[tick]
                )
                for household in households
            }
            for tick in range(ticks)
        ]

    environments = {
        household.profile.household_id: HouseholdEnvironment(
            profile=household.profile,
            demand_kwh=household.demand_kwh,
            solar_kwh=household.solar_kwh,
        )
        for household in households
    }
    return [
        {
            household_id: environment.step().net_position_kwh
            for household_id, environment in environments.items()
        }
        for _ in range(ticks)
    ]


def _grid_imports_kwh(
    state: MarketState, metered_net_kwh: dict[str, float]
) -> tuple[float, float]:
    """Community grid import for one tick, in kWh: (under P2P, with no marketplace).

    With no marketplace every household imports its whole deficit. Under P2P it
    imports only what the market did not supply — its metered position adjusted by
    the energy it actually bought and sold. Surpluses are excluded from both figures:
    exporting to the grid is not load, and peak load is what this feeds.
    """
    net_sold_kwh: dict[str, float] = {}
    for trade in state.trades:
        net_sold_kwh[trade.seller_id] = (
            net_sold_kwh.get(trade.seller_id, 0.0) + trade.quantity_kwh
        )
        net_sold_kwh[trade.buyer_id] = (
            net_sold_kwh.get(trade.buyer_id, 0.0) - trade.quantity_kwh
        )

    p2p_import_kwh = 0.0
    grid_only_import_kwh = 0.0
    for household_id, metered_kwh in metered_net_kwh.items():
        grid_only_import_kwh += max(0.0, -metered_kwh)
        imbalance_kwh = metered_kwh - net_sold_kwh.get(household_id, 0.0)
        p2p_import_kwh += max(0.0, -imbalance_kwh)
    return p2p_import_kwh, grid_only_import_kwh


def run_pipeline(
    households: Sequence[HouseholdSeries] | None = None,
    num_ticks: int | None = None,
    window: int = 4,
    run_id: str | None = None,
    strategy: TradingStrategy | None = None,
    strategy_name: str = STRATEGY_NAME,
    store: RunStore | None = None,
    db_path: str | Path | None = None,
    use_battery: bool = False,
    grid_co2_g_per_kwh: float = IE_GRID_CO2_G_PER_KWH,
) -> PipelineResult:
    """Run the full pipeline and return everything it produced.

    Per tick: forecast every household from history only, ask the strategy for an
    order, clear the book, settle against the meter, persist. Households default to
    :func:`demo_households`; a ``store`` (or a ``db_path`` to open one) persists the
    run, and omitting both runs in memory only.
    """
    households = list(households) if households is not None else demo_households()
    ticks = _tick_count(households, num_ticks)
    run_id = run_id or f"run_{strategy_name}_{datetime.now():%Y%m%d_%H%M%S}"

    owns_store = store is None and db_path is not None
    if owns_store:
        store = RunStore(db_path)  # type: ignore[arg-type]

    trader = strategy or RuleBasedTrader()
    forecaster = NaiveForecaster(window=window)
    profiles = {h.profile.household_id: h.profile for h in households}
    metered_per_tick = _metered_net_positions(households, ticks, use_battery)

    forecasts: list[ForecastOutput] = []
    decisions: list[AgentDecision] = []
    market_states: list[MarketState] = []
    p2p_cost_eur = dict.fromkeys(profiles, 0.0)
    grid_cost_eur = dict.fromkeys(profiles, 0.0)
    traded_kwh = 0.0
    p2p_imports_kwh: list[float] = []
    grid_only_imports_kwh: list[float] = []

    try:
        if store is not None:
            store.save_household_profiles(run_id, profiles.values())

        for tick in range(ticks):
            timestamp = _timestamp(tick)
            metered_net_kwh = metered_per_tick[tick]

            tick_forecasts = [
                forecaster.forecast(
                    household_id=household.profile.household_id,
                    tick=tick,
                    timestamp=timestamp,
                    demand_history_kwh=household.demand_kwh[:tick],
                    solar_history_kwh=household.solar_kwh[:tick],
                )
                for household in households
            ]
            orders = [
                order
                for forecast, household in zip(tick_forecasts, households, strict=True)
                if (order := trader.decide(forecast, household.profile)) is not None
            ]
            state = clear_tick(tick=tick, timestamp=timestamp, orders=orders)

            for household_id, settlement in settle_tick(
                state, profiles, actual_net_position_kwh=metered_net_kwh
            ).items():
                p2p_cost_eur[household_id] += settlement.p2p_cost_eur
                grid_cost_eur[household_id] += settlement.grid_only_cost_eur

            traded_kwh += sum(trade.quantity_kwh for trade in state.trades)
            p2p_import_kwh, grid_only_import_kwh = _grid_imports_kwh(state, metered_net_kwh)
            p2p_imports_kwh.append(p2p_import_kwh)
            grid_only_imports_kwh.append(grid_only_import_kwh)

            forecasts.extend(tick_forecasts)
            decisions.extend(orders)
            market_states.append(state)

            if store is not None:
                store.save_forecasts(run_id, tick_forecasts)
                store.save_decisions(run_id, orders)
                store.save_market_state(run_id, state)

        outcomes = compute_household_outcomes(p2p_cost_eur, grid_cost_eur)
        summary = compute_run_summary(
            run_id=run_id,
            strategy_name=strategy_name,
            outcomes=outcomes,
            num_ticks=ticks,
            traded_kwh=traded_kwh,
            p2p_grid_import_kwh_per_tick=p2p_imports_kwh,
            grid_only_import_kwh_per_tick=grid_only_imports_kwh,
            grid_co2_g_per_kwh=grid_co2_g_per_kwh,
        )
        if store is not None:
            store.save_household_outcomes(run_id, outcomes)
            store.save_run_summary(summary)
    finally:
        if owns_store and store is not None:
            store.close()

    return PipelineResult(
        run_id=run_id,
        strategy_name=strategy_name,
        forecasts=forecasts,
        decisions=decisions,
        market_states=market_states,
        outcomes=outcomes,
        summary=summary,
    )


def main() -> None:
    """Run the full pipeline on demo households, persist it, print the headline numbers.

    Persisting is the point of the default run: the dashboard renders what was
    stored, never a run it recomputes itself.
    """
    households = demo_households()
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = run_pipeline(households, db_path=DEFAULT_DB_PATH)
    summary = result.summary
    assert summary is not None  # run_pipeline always summarises a non-empty run

    print(
        f"Orchestrator: {summary.num_households} households x {summary.num_ticks} ticks "
        f"-> {len(result.forecasts)} forecasts, {len(result.decisions)} decisions, "
        f"{len(result.trades)} trades"
    )
    print(f"  total savings:       EUR {summary.total_savings_eur:.2f}")
    print(f"  avg savings:         {summary.avg_savings_pct:.1f}%")
    print(f"  peak load reduction: {summary.peak_load_reduction_pct:.1f}%")
    print(f"  CO2 avoided:         {summary.co2_avoided_kg:.2f} kg")
    for outcome in result.outcomes:
        print(
            f"    {outcome.household_id}: EUR {outcome.savings_eur:+.2f} "
            f"({outcome.savings_pct:+.1f}%)"
        )
    print(f"  persisted run {summary.run_id} to {DEFAULT_DB_PATH}")
    print("  view it with: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
