"""Run data access: the one place the dashboard talks to persistence.

The dashboard reads history through ``dashboard.persistence.RunStore`` and never
recomputes a run to draw it — what is on screen is what was persisted. That
module is owned elsewhere and lands on its own branch, so this file codes
against its published signatures and falls back to an in-memory stand-in when
the import fails; see ``_fixture_store``.

RunStore methods used here (all keyed by run_id):

    list_runs()                     -> list[str]
    load_run_summary(run_id)        -> RunSummary | None
    load_household_outcomes(run_id) -> list[HouseholdOutcome]
    load_household_profiles(run_id) -> list[HouseholdProfile]
    load_trades(run_id)             -> list[TradeEvent]
    load_market_states(run_id)      -> list[MarketState]
    load_forecasts(run_id)          -> list[ForecastOutput]
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from shared.schemas import (
    AgentRole,
    ForecastOutput,
    HouseholdOutcome,
    HouseholdProfile,
    MarketState,
    RunSummary,
    TradeEvent,
)

#: SQLite file the orchestrator persists runs to. Override with GRIDPEER_DB when
#: pointing the dashboard at a run produced somewhere else.
DB_PATH = Path(os.environ.get("GRIDPEER_DB", "data/gridpeer.db"))

#: Used only when a run has no stored HouseholdProfile rows to read tariffs from.
DEFAULT_IMPORT_TARIFF_EUR_PER_KWH = 0.25
DEFAULT_EXPORT_TARIFF_EUR_PER_KWH = 0.07


class RunStoreLike(Protocol):
    """Structural type for the persistence API this dashboard depends on."""

    def list_runs(self) -> list[str]: ...
    def load_run_summary(self, run_id: str) -> RunSummary | None: ...
    def load_household_outcomes(self, run_id: str) -> list[HouseholdOutcome]: ...
    def load_household_profiles(self, run_id: str) -> list[HouseholdProfile]: ...
    def load_trades(self, run_id: str) -> list[TradeEvent]: ...
    def load_market_states(self, run_id: str) -> list[MarketState]: ...
    def load_forecasts(self, run_id: str) -> list[ForecastOutput]: ...


@dataclass(frozen=True)
class RunData:
    """Everything the views need for one run, fetched once per render."""

    summary: RunSummary
    outcomes: list[HouseholdOutcome]
    trades: list[TradeEvent]
    market_states: list[MarketState]
    forecasts: list[ForecastOutput]
    tariffs_eur_per_kwh: dict[str, tuple[float, float]]

    @property
    def household_ids(self) -> list[str]:
        """Sorted household ids in this run, for stable colour assignment."""
        return sorted({o.household_id for o in self.outcomes})


def open_store() -> tuple[RunStoreLike, bool]:
    """Return the persistence store and whether it is the stand-in fixture."""
    try:
        from dashboard.persistence import RunStore
    except ImportError:
        return _fixture_store(), True  # DELETE THIS LINE once persistence.py lands
    return RunStore(DB_PATH), False


def list_summaries(store: RunStoreLike) -> list[RunSummary]:
    """Every summarised run, newest first, skipping runs that were never summarised."""
    summaries = (store.load_run_summary(run_id) for run_id in store.list_runs())
    return [s for s in summaries if s is not None]


def load_run(store: RunStoreLike, summary: RunSummary) -> RunData:
    """Fetch one run's outcomes, trades, market states and forecasts."""
    run_id = summary.run_id
    return RunData(
        summary=summary,
        outcomes=store.load_household_outcomes(run_id),
        trades=store.load_trades(run_id),
        market_states=store.load_market_states(run_id),
        forecasts=store.load_forecasts(run_id),
        tariffs_eur_per_kwh={
            p.household_id: (
                p.grid_import_tariff_eur_per_kwh,
                p.grid_export_tariff_eur_per_kwh,
            )
            for p in store.load_household_profiles(run_id)
        },
    )


def tariff_for(
    tariffs: dict[str, tuple[float, float]], household_id: str
) -> tuple[float, float]:
    """(import, export) tariff in EUR/kWh for one household, or the defaults."""
    return tariffs.get(
        household_id,
        (DEFAULT_IMPORT_TARIFF_EUR_PER_KWH, DEFAULT_EXPORT_TARIFF_EUR_PER_KWH),
    )


# ---------------------------------------------------------------------------
# Stand-in data — delete this whole section when dashboard/persistence.py lands
# ---------------------------------------------------------------------------

_FIXTURE_EPOCH = datetime(2026, 1, 1, 0, 0)
_FIXTURE_TICKS = 96  # two days at half-hourly resolution
_FIXTURE_ROLES = (
    AgentRole.PROSUMER,
    AgentRole.PURE_CONSUMER,
    AgentRole.PROSUMER,
    AgentRole.PURE_PRODUCER,
)
_FIXTURE_HOUSEHOLDS = tuple(f"hh_{i:03d}" for i in range(len(_FIXTURE_ROLES)))


def _fixture_store() -> RunStoreLike:
    """In-memory RunStore stand-in so the views render before persistence lands.

    Shapes and magnitudes are plausible, not meaningful — nothing here should
    ever reach a slide. It exists to exercise the charts, the run selector and
    the comparison tab while the real store is built on another branch.
    """

    class _FixtureStore:
        """Satisfies RunStoreLike from hard-coded synthetic runs."""

        def __init__(self) -> None:
            self._runs = {
                "run_rule_based_baseline_fixture": _fixture_run(
                    "run_rule_based_baseline_fixture", "rule_based_baseline", 1.0
                ),
                "run_ppo_v1_fixture": _fixture_run("run_ppo_v1_fixture", "ppo_v1", 1.28),
            }

        def list_runs(self) -> list[str]:
            """Stored run ids, newest first."""
            return list(self._runs)

        def load_run_summary(self, run_id: str) -> RunSummary | None:
            """One run's headline rollup."""
            run = self._runs.get(run_id)
            return None if run is None else run.summary

        def load_household_outcomes(self, run_id: str) -> list[HouseholdOutcome]:
            """Per-household cumulative outcomes for one run."""
            return self._runs[run_id].outcomes

        def load_household_profiles(self, run_id: str) -> list[HouseholdProfile]:
            """Household identities, which carry the grid tariffs."""
            return _fixture_profiles()

        def load_trades(self, run_id: str) -> list[TradeEvent]:
            """Every cleared trade in one run."""
            return self._runs[run_id].trades

        def load_market_states(self, run_id: str) -> list[MarketState]:
            """Per-tick market snapshots for one run."""
            return self._runs[run_id].market_states

        def load_forecasts(self, run_id: str) -> list[ForecastOutput]:
            """The ForecastOutput stream the run was driven by."""
            return self._runs[run_id].forecasts

    return _FixtureStore()


def _fixture_profiles() -> list[HouseholdProfile]:
    """Four demo households matching the orchestrator's roles and tariffs."""
    profiles: list[HouseholdProfile] = []
    for household_id, role in zip(_FIXTURE_HOUSEHOLDS, _FIXTURE_ROLES, strict=True):
        has_solar = role is not AgentRole.PURE_CONSUMER
        profiles.append(
            HouseholdProfile(
                household_id=household_id,
                role=role,
                has_solar=has_solar,
                solar_capacity_kw=3.0 if has_solar else 0.0,
                battery_capacity_kwh=5.0 if has_solar else 0.0,
                grid_export_tariff_eur_per_kwh=DEFAULT_EXPORT_TARIFF_EUR_PER_KWH,
                grid_import_tariff_eur_per_kwh=DEFAULT_IMPORT_TARIFF_EUR_PER_KWH,
            )
        )
    return profiles


def _fixture_untraded_cost_eur() -> dict[str, float]:
    """Each household's whole-run grid import bill in EUR, before any trading."""
    costs: dict[str, float] = dict.fromkeys(_FIXTURE_HOUSEHOLDS, 0.0)
    for tick in range(_FIXTURE_TICKS):
        hour = (tick % 48) * 0.5
        for household_id in _FIXTURE_HOUSEHOLDS:
            net = _fixture_demand_kwh(household_id, hour) - _fixture_solar_kwh(
                household_id, hour
            )
            if net > 0:
                costs[household_id] += net * DEFAULT_IMPORT_TARIFF_EUR_PER_KWH
    return costs


@dataclass(frozen=True)
class _FixtureRun:
    """One synthetic run's record sets."""

    summary: RunSummary
    outcomes: list[HouseholdOutcome]
    trades: list[TradeEvent]
    market_states: list[MarketState]
    forecasts: list[ForecastOutput]


def _fixture_run(run_id: str, strategy_name: str, uplift: float) -> _FixtureRun:
    """Build one synthetic run; ``uplift`` scales how well the strategy trades."""
    trades = _fixture_trades(run_id, uplift)
    outcomes = _fixture_outcomes(trades)
    total_kwh = sum(t.quantity_kwh for t in trades)
    grid_total = sum(abs(o.total_cost_eur_grid_baseline) for o in outcomes) or 1.0
    total_savings = sum(o.savings_eur for o in outcomes)
    return _FixtureRun(
        summary=RunSummary(
            run_id=run_id,
            strategy_name=strategy_name,
            num_households=len(_FIXTURE_HOUSEHOLDS),
            num_ticks=_FIXTURE_TICKS,
            total_savings_eur=round(total_savings, 2),
            avg_savings_pct=round(100.0 * total_savings / grid_total, 2),
            peak_load_reduction_pct=round(11.4 * uplift, 2),
            co2_avoided_kg=round(total_kwh * 0.33, 2),
        ),
        outcomes=outcomes,
        trades=trades,
        market_states=_fixture_market_states(trades),
        forecasts=_fixture_forecasts(),
    )


def _fixture_solar_kwh(household_id: str, hour: float) -> float:
    """Synthetic half-hourly solar output for one household, kWh."""
    if household_id == "hh_001" or not 6.0 <= hour <= 20.0:
        return 0.0
    return round(1.5 * math.exp(-((hour - 13.0) ** 2) / 8.0), 4)


def _fixture_demand_kwh(household_id: str, hour: float) -> float:
    """Synthetic half-hourly demand for one household, kWh."""
    if household_id == "hh_003":  # pure producer, negligible demand
        return 0.15
    morning = math.exp(-((hour - 8.0) ** 2) / 2.0)
    evening = math.exp(-((hour - 19.0) ** 2) / 2.0)
    return round(0.3 + 1.2 * (morning + evening), 4)


def _fixture_forecasts() -> list[ForecastOutput]:
    """A ForecastOutput per household per tick, shaped like the naive baseline."""
    stream: list[ForecastOutput] = []
    for tick in range(_FIXTURE_TICKS):
        hour = (tick % 48) * 0.5
        timestamp = _FIXTURE_EPOCH + timedelta(minutes=30 * tick)
        for household_id in _FIXTURE_HOUSEHOLDS:
            demand = _fixture_demand_kwh(household_id, hour)
            solar = _fixture_solar_kwh(household_id, hour)
            stream.append(
                ForecastOutput(
                    household_id=household_id,
                    tick=tick,
                    timestamp=timestamp,
                    predicted_demand_kwh=demand,
                    predicted_solar_generation_kwh=solar,
                    predicted_net_position_kwh=round(solar - demand, 4),
                    confidence=round(0.55 + 0.35 * min(tick, 8) / 8.0, 3),
                )
            )
    return stream


def _fixture_trades(run_id: str, uplift: float) -> list[TradeEvent]:
    """Synthetic cleared trades: daytime solar surplus sold to the consumer."""
    trades: list[TradeEvent] = []
    for tick in range(_FIXTURE_TICKS):
        hour = (tick % 48) * 0.5
        if not 7.0 <= hour <= 19.0:
            continue
        midday = math.exp(-((hour - 13.0) ** 2) / 9.0)
        quantity = round(0.9 * midday * uplift, 4)
        if quantity < 0.02:
            continue
        price = round(0.11 + 0.05 * (1.0 - midday), 4)
        for seller in ("hh_000", "hh_002", "hh_003"):
            trades.append(
                TradeEvent(
                    trade_id=f"{run_id}-t{tick}-{seller}",
                    tick=tick,
                    timestamp=_FIXTURE_EPOCH + timedelta(minutes=30 * tick),
                    buyer_id="hh_001",
                    seller_id=seller,
                    quantity_kwh=quantity,
                    clearing_price_eur_per_kwh=price,
                )
            )
    return trades


def _fixture_market_states(trades: list[TradeEvent]) -> list[MarketState]:
    """One MarketState per tick that cleared, carrying that tick's trades."""
    by_tick: dict[int, list[TradeEvent]] = {}
    for trade in trades:
        by_tick.setdefault(trade.tick, []).append(trade)
    return [
        MarketState(
            tick=tick,
            timestamp=_FIXTURE_EPOCH + timedelta(minutes=30 * tick),
            trades=tick_trades,
            unmatched_buy_orders=[],
            unmatched_sell_orders=[],
            clearing_price_eur_per_kwh=tick_trades[0].clearing_price_eur_per_kwh,
        )
        for tick, tick_trades in sorted(by_tick.items())
    ]


def _fixture_outcomes(trades: list[TradeEvent]) -> list[HouseholdOutcome]:
    """Roll synthetic trades up into per-household cost/savings outcomes.

    Both bases start from the household's whole-run grid bill, not just the part
    it traded — otherwise the denominator is only the traded slice and the
    savings percentage comes out implausibly high for stand-in data.
    """
    untraded = _fixture_untraded_cost_eur()
    p2p: dict[str, float] = dict(untraded)
    grid: dict[str, float] = dict(untraded)
    for trade in trades:
        value = trade.quantity_kwh * trade.clearing_price_eur_per_kwh
        p2p[trade.buyer_id] += value
        p2p[trade.seller_id] -= value
        grid[trade.buyer_id] += trade.quantity_kwh * DEFAULT_IMPORT_TARIFF_EUR_PER_KWH
        grid[trade.seller_id] -= trade.quantity_kwh * DEFAULT_EXPORT_TARIFF_EUR_PER_KWH
    outcomes: list[HouseholdOutcome] = []
    for household_id in _FIXTURE_HOUSEHOLDS:
        savings = grid[household_id] - p2p[household_id]
        denominator = abs(grid[household_id]) or 1.0
        outcomes.append(
            HouseholdOutcome(
                household_id=household_id,
                total_cost_eur_p2p=round(p2p[household_id], 2),
                total_cost_eur_grid_baseline=round(grid[household_id], 2),
                savings_eur=round(savings, 2),
                savings_pct=round(100.0 * savings / denominator, 2),
            )
        )
    return outcomes
