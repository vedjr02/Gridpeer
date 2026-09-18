"""SQLite persistence for a pipeline run: one table per shared model, keyed by run_id.

Build-order step 2 (see dashboard/AGENTS.md). This exists so the dashboard can be
re-opened without re-running the simulation, and so two runs — baseline versus a
learned policy — can be compared after the fact instead of side by side in memory.
It is deliberately a thin store, not an ORM: every row is a shared contract in,
the same shared contract out, and nothing here interprets the numbers.

Schema, one table per model in shared/schemas.py:

    households          HouseholdProfile   PK (run_id, household_id)
    forecasts           ForecastOutput     PK (run_id, household_id, tick)
    decisions           AgentDecision      PK (run_id, household_id, tick)
    trades              TradeEvent         PK (run_id, trade_id)
    market_states       MarketState        PK (run_id, tick)
    household_outcomes  HouseholdOutcome   PK (run_id, household_id)
    run_summaries       RunSummary         PK (run_id)

Everything is keyed by run_id because the comparison the project is built to make
needs two runs in one database. Writes are idempotent (INSERT OR REPLACE), so
re-running a tick overwrites it rather than silently doubling the history.

Two storage choices worth knowing about:

- Timestamps are ISO-8601 text, not Unix epochs. SQLite has no datetime type
  either way, and text sorts chronologically and stays readable when somebody
  opens the file with the sqlite3 CLI to work out what went wrong.
- ``MarketState.trades`` is not duplicated into the market_states table — it is
  reconstructed from the trades table on load, so a trade has exactly one row and
  cannot disagree with itself. The unmatched order books have no table of their
  own (they are AgentDecisions that never cleared, not decisions the agent made)
  so those are stored as JSON on the market_states row.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType

from shared.schemas import (
    AgentDecision,
    AgentRole,
    ForecastOutput,
    HouseholdOutcome,
    HouseholdProfile,
    MarketState,
    OrderSide,
    RunSummary,
    TradeEvent,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS households (
    run_id                          TEXT NOT NULL,
    household_id                    TEXT NOT NULL,
    role                            TEXT NOT NULL,
    has_solar                       INTEGER NOT NULL,
    solar_capacity_kw               REAL NOT NULL,
    battery_capacity_kwh            REAL NOT NULL,
    grid_export_tariff_eur_per_kwh  REAL NOT NULL,
    grid_import_tariff_eur_per_kwh  REAL NOT NULL,
    PRIMARY KEY (run_id, household_id)
);

CREATE TABLE IF NOT EXISTS forecasts (
    run_id                          TEXT NOT NULL,
    household_id                    TEXT NOT NULL,
    tick                            INTEGER NOT NULL,
    timestamp                       TEXT NOT NULL,
    predicted_demand_kwh            REAL NOT NULL,
    predicted_solar_generation_kwh  REAL NOT NULL,
    predicted_net_position_kwh      REAL NOT NULL,
    confidence                      REAL NOT NULL,
    PRIMARY KEY (run_id, household_id, tick)
);

CREATE TABLE IF NOT EXISTS decisions (
    run_id                     TEXT NOT NULL,
    household_id               TEXT NOT NULL,
    tick                       INTEGER NOT NULL,
    side                       TEXT NOT NULL,
    quantity_kwh               REAL NOT NULL,
    limit_price_eur_per_kwh    REAL NOT NULL,
    strategy_name              TEXT NOT NULL,
    PRIMARY KEY (run_id, household_id, tick)
);

CREATE TABLE IF NOT EXISTS trades (
    run_id                       TEXT NOT NULL,
    trade_id                     TEXT NOT NULL,
    tick                         INTEGER NOT NULL,
    timestamp                    TEXT NOT NULL,
    buyer_id                     TEXT NOT NULL,
    seller_id                    TEXT NOT NULL,
    quantity_kwh                 REAL NOT NULL,
    clearing_price_eur_per_kwh   REAL NOT NULL,
    PRIMARY KEY (run_id, trade_id)
);
CREATE INDEX IF NOT EXISTS idx_trades_tick ON trades (run_id, tick);

CREATE TABLE IF NOT EXISTS market_states (
    run_id                       TEXT NOT NULL,
    tick                         INTEGER NOT NULL,
    timestamp                    TEXT NOT NULL,
    clearing_price_eur_per_kwh   REAL,
    unmatched_buy_orders_json    TEXT NOT NULL,
    unmatched_sell_orders_json   TEXT NOT NULL,
    PRIMARY KEY (run_id, tick)
);

CREATE TABLE IF NOT EXISTS household_outcomes (
    run_id                        TEXT NOT NULL,
    household_id                  TEXT NOT NULL,
    total_cost_eur_p2p            REAL NOT NULL,
    total_cost_eur_grid_baseline  REAL NOT NULL,
    savings_eur                   REAL NOT NULL,
    savings_pct                   REAL NOT NULL,
    PRIMARY KEY (run_id, household_id)
);

CREATE TABLE IF NOT EXISTS run_summaries (
    run_id                   TEXT PRIMARY KEY,
    strategy_name            TEXT NOT NULL,
    num_households           INTEGER NOT NULL,
    num_ticks                INTEGER NOT NULL,
    total_savings_eur        REAL NOT NULL,
    avg_savings_pct          REAL NOT NULL,
    peak_load_reduction_pct  REAL NOT NULL,
    co2_avoided_kg           REAL NOT NULL,
    created_at               TEXT NOT NULL
);
"""


def _orders_to_json(orders: Sequence[AgentDecision]) -> str:
    """Serialise an unmatched order book to a JSON array for a market_states row."""
    return json.dumps([order.model_dump(mode="json") for order in orders])


def _orders_from_json(payload: str) -> list[AgentDecision]:
    """Rebuild an unmatched order book from a market_states row."""
    return [AgentDecision.model_validate(item) for item in json.loads(payload)]


class RunStore:
    """SQLite-backed history for one or more pipeline runs.

    Usable as a context manager, which is the intended way — it closes the
    connection on exit::

        with RunStore("gridpeer.db") as store:
            store.save_trades(run_id, state.trades)

    Pass ``":memory:"`` (the default) for a throwaway store; tests and the
    Streamlit app both rely on that being cheap.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        """db_path: SQLite file to open or create; ":memory:" for an ephemeral store."""
        self.db_path = str(db_path)
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> RunStore:
        """Enter the context manager; the store is already open."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the connection on the way out, committed or not."""
        self.close()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()

    # -- HouseholdProfile --------------------------------------------------

    def save_household_profiles(
        self, run_id: str, profiles: Iterable[HouseholdProfile]
    ) -> None:
        """Persist the households taking part in ``run_id``."""
        rows = [
            (
                run_id,
                p.household_id,
                p.role.value,
                int(p.has_solar),
                p.solar_capacity_kw,
                p.battery_capacity_kwh,
                p.grid_export_tariff_eur_per_kwh,
                p.grid_import_tariff_eur_per_kwh,
            )
            for p in profiles
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO households VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        self._connection.commit()

    def load_household_profiles(self, run_id: str) -> list[HouseholdProfile]:
        """Every household in ``run_id``, ordered by household_id."""
        rows = self._connection.execute(
            "SELECT * FROM households WHERE run_id = ? ORDER BY household_id", (run_id,)
        ).fetchall()
        return [
            HouseholdProfile(
                household_id=row["household_id"],
                role=AgentRole(row["role"]),
                has_solar=bool(row["has_solar"]),
                solar_capacity_kw=row["solar_capacity_kw"],
                battery_capacity_kwh=row["battery_capacity_kwh"],
                grid_export_tariff_eur_per_kwh=row["grid_export_tariff_eur_per_kwh"],
                grid_import_tariff_eur_per_kwh=row["grid_import_tariff_eur_per_kwh"],
            )
            for row in rows
        ]

    # -- ForecastOutput ----------------------------------------------------

    def save_forecasts(self, run_id: str, forecasts: Iterable[ForecastOutput]) -> None:
        """Persist a tick's (or a whole run's) ForecastOutput stream."""
        rows = [
            (
                run_id,
                f.household_id,
                f.tick,
                f.timestamp.isoformat(),
                f.predicted_demand_kwh,
                f.predicted_solar_generation_kwh,
                f.predicted_net_position_kwh,
                f.confidence,
            )
            for f in forecasts
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        self._connection.commit()

    def load_forecasts(self, run_id: str) -> list[ForecastOutput]:
        """Every forecast in ``run_id``, in tick then household order."""
        rows = self._connection.execute(
            "SELECT * FROM forecasts WHERE run_id = ? ORDER BY tick, household_id",
            (run_id,),
        ).fetchall()
        return [
            ForecastOutput(
                household_id=row["household_id"],
                tick=row["tick"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                predicted_demand_kwh=row["predicted_demand_kwh"],
                predicted_solar_generation_kwh=row["predicted_solar_generation_kwh"],
                predicted_net_position_kwh=row["predicted_net_position_kwh"],
                confidence=row["confidence"],
            )
            for row in rows
        ]

    # -- AgentDecision -----------------------------------------------------

    def save_decisions(self, run_id: str, decisions: Iterable[AgentDecision]) -> None:
        """Persist a tick's order book as submitted by the agents."""
        rows = [
            (
                run_id,
                d.household_id,
                d.tick,
                d.side.value,
                d.quantity_kwh,
                d.limit_price_eur_per_kwh,
                d.strategy_name,
            )
            for d in decisions
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?)", rows
        )
        self._connection.commit()

    def load_decisions(self, run_id: str) -> list[AgentDecision]:
        """Every decision in ``run_id``, in tick then household order."""
        rows = self._connection.execute(
            "SELECT * FROM decisions WHERE run_id = ? ORDER BY tick, household_id",
            (run_id,),
        ).fetchall()
        return [_decision_from_row(row) for row in rows]

    # -- TradeEvent --------------------------------------------------------

    def save_trades(self, run_id: str, trades: Iterable[TradeEvent]) -> None:
        """Persist cleared trades."""
        rows = [
            (
                run_id,
                t.trade_id,
                t.tick,
                t.timestamp.isoformat(),
                t.buyer_id,
                t.seller_id,
                t.quantity_kwh,
                t.clearing_price_eur_per_kwh,
            )
            for t in trades
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        self._connection.commit()

    def load_trades(self, run_id: str, tick: int | None = None) -> list[TradeEvent]:
        """Trades in ``run_id``, all of them or just one tick's, in trade_id order."""
        if tick is None:
            rows = self._connection.execute(
                "SELECT * FROM trades WHERE run_id = ? ORDER BY tick, trade_id", (run_id,)
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM trades WHERE run_id = ? AND tick = ? ORDER BY trade_id",
                (run_id, tick),
            ).fetchall()
        return [_trade_from_row(row) for row in rows]

    # -- MarketState -------------------------------------------------------

    def save_market_state(self, run_id: str, state: MarketState) -> None:
        """Persist one cleared tick, including its trades.

        The trades go to the trades table rather than onto this row, so
        :meth:`load_market_states` can rebuild them and there is only ever one
        copy of a trade in the database.
        """
        self.save_trades(run_id, state.trades)
        self._connection.execute(
            "INSERT OR REPLACE INTO market_states VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                state.tick,
                state.timestamp.isoformat(),
                state.clearing_price_eur_per_kwh,
                _orders_to_json(state.unmatched_buy_orders),
                _orders_to_json(state.unmatched_sell_orders),
            ),
        )
        self._connection.commit()

    def load_market_states(self, run_id: str) -> list[MarketState]:
        """Every tick of ``run_id`` in tick order, trades rejoined from the trades table."""
        rows = self._connection.execute(
            "SELECT * FROM market_states WHERE run_id = ? ORDER BY tick", (run_id,)
        ).fetchall()
        trades_by_tick: dict[int, list[TradeEvent]] = {}
        for trade in self.load_trades(run_id):
            trades_by_tick.setdefault(trade.tick, []).append(trade)
        return [
            MarketState(
                tick=row["tick"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                trades=trades_by_tick.get(row["tick"], []),
                unmatched_buy_orders=_orders_from_json(row["unmatched_buy_orders_json"]),
                unmatched_sell_orders=_orders_from_json(row["unmatched_sell_orders_json"]),
                clearing_price_eur_per_kwh=row["clearing_price_eur_per_kwh"],
            )
            for row in rows
        ]

    # -- HouseholdOutcome / RunSummary -------------------------------------

    def save_household_outcomes(
        self, run_id: str, outcomes: Iterable[HouseholdOutcome]
    ) -> None:
        """Persist the per-household rollups for ``run_id``."""
        rows = [
            (
                run_id,
                o.household_id,
                o.total_cost_eur_p2p,
                o.total_cost_eur_grid_baseline,
                o.savings_eur,
                o.savings_pct,
            )
            for o in outcomes
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO household_outcomes VALUES (?, ?, ?, ?, ?, ?)", rows
        )
        self._connection.commit()

    def load_household_outcomes(self, run_id: str) -> list[HouseholdOutcome]:
        """Per-household rollups for ``run_id``, ordered by household_id."""
        rows = self._connection.execute(
            "SELECT * FROM household_outcomes WHERE run_id = ? ORDER BY household_id",
            (run_id,),
        ).fetchall()
        return [
            HouseholdOutcome(
                household_id=row["household_id"],
                total_cost_eur_p2p=row["total_cost_eur_p2p"],
                total_cost_eur_grid_baseline=row["total_cost_eur_grid_baseline"],
                savings_eur=row["savings_eur"],
                savings_pct=row["savings_pct"],
            )
            for row in rows
        ]

    def save_run_summary(self, summary: RunSummary) -> None:
        """Persist the headline result. ``summary.run_id`` is the key — no separate arg."""
        self._connection.execute(
            "INSERT OR REPLACE INTO run_summaries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                summary.run_id,
                summary.strategy_name,
                summary.num_households,
                summary.num_ticks,
                summary.total_savings_eur,
                summary.avg_savings_pct,
                summary.peak_load_reduction_pct,
                summary.co2_avoided_kg,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        self._connection.commit()

    def load_run_summary(self, run_id: str) -> RunSummary | None:
        """The headline result for ``run_id``, or None if that run was never summarised."""
        row = self._connection.execute(
            "SELECT * FROM run_summaries WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return RunSummary(
            run_id=row["run_id"],
            strategy_name=row["strategy_name"],
            num_households=row["num_households"],
            num_ticks=row["num_ticks"],
            total_savings_eur=row["total_savings_eur"],
            avg_savings_pct=row["avg_savings_pct"],
            peak_load_reduction_pct=row["peak_load_reduction_pct"],
            co2_avoided_kg=row["co2_avoided_kg"],
        )

    # -- run-level ---------------------------------------------------------

    def list_runs(self) -> list[str]:
        """Summarised run_ids, newest first — what the dashboard's run picker shows."""
        rows = self._connection.execute(
            "SELECT run_id FROM run_summaries ORDER BY created_at DESC, run_id DESC"
        ).fetchall()
        return [row["run_id"] for row in rows]

    def delete_run(self, run_id: str) -> None:
        """Remove every row belonging to ``run_id``, so a re-run starts clean."""
        for table in (
            "households",
            "forecasts",
            "decisions",
            "trades",
            "market_states",
            "household_outcomes",
            "run_summaries",
        ):
            self._connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        self._connection.commit()


def _decision_from_row(row: sqlite3.Row) -> AgentDecision:
    """Rebuild an AgentDecision from a decisions row."""
    return AgentDecision(
        household_id=row["household_id"],
        tick=row["tick"],
        side=OrderSide(row["side"]),
        quantity_kwh=row["quantity_kwh"],
        limit_price_eur_per_kwh=row["limit_price_eur_per_kwh"],
        strategy_name=row["strategy_name"],
    )


def _trade_from_row(row: sqlite3.Row) -> TradeEvent:
    """Rebuild a TradeEvent from a trades row."""
    return TradeEvent(
        trade_id=row["trade_id"],
        tick=row["tick"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        buyer_id=row["buyer_id"],
        seller_id=row["seller_id"],
        quantity_kwh=row["quantity_kwh"],
        clearing_price_eur_per_kwh=row["clearing_price_eur_per_kwh"],
    )
