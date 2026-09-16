"""
Shared data contracts for GridPeer.

This is the single source of truth for how the four modules (simulation,
agents, forecasting, dashboard) talk to each other. No module should pass
bare dicts across module boundaries — construct and consume these models.

SCHEMA_VERSION: bump this on any breaking change. See CONTRIBUTING.md for
the sign-off process required before changing anything in this file.

Units are explicit throughout:
    - energy: kWh (kilowatt-hours)
    - power: kW (kilowatts)
    - price: EUR per kWh
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    """Whether an agent is offering to buy or sell energy in a given tick."""

    BUY = "buy"
    SELL = "sell"


class AgentRole(str, Enum):
    """Household archetype. Determines available strategies, not identity."""

    PURE_CONSUMER = "pure_consumer"       # no generation, only buys
    PROSUMER = "prosumer"                 # has solar + battery, buys and sells
    PURE_PRODUCER = "pure_producer"       # generation only, no significant demand


# ---------------------------------------------------------------------------
# Household state
# ---------------------------------------------------------------------------

class HouseholdProfile(BaseModel):
    """Static + slowly-changing attributes of a household in the simulation.

    This is not re-sent every tick — it's the household's "identity."
    Fast-changing state (current demand, current battery level) belongs in
    the tick-level models below, not here.
    """

    household_id: str
    role: AgentRole
    has_solar: bool
    solar_capacity_kw: float = Field(ge=0, description="Peak solar generation capacity, kW")
    battery_capacity_kwh: float = Field(ge=0, description="Total battery storage capacity, kWh")
    grid_export_tariff_eur_per_kwh: float = Field(
        ge=0, description="Fallback tariff if the household just exports to the grid instead of trading"
    )
    grid_import_tariff_eur_per_kwh: float = Field(
        ge=0, description="Fallback tariff if the household just imports from the grid instead of trading"
    )


# ---------------------------------------------------------------------------
# Forecasting -> Agents
# ---------------------------------------------------------------------------

class ForecastOutput(BaseModel):
    """One household's forecast for a given simulation tick.

    Produced by: forecasting/
    Consumed by: agents/ (to decide bids/asks), dashboard/ (for display)
    """

    household_id: str
    tick: int
    timestamp: datetime
    predicted_demand_kwh: float = Field(ge=0)
    predicted_solar_generation_kwh: float = Field(ge=0)
    predicted_net_position_kwh: float = Field(
        description="predicted_solar_generation_kwh - predicted_demand_kwh. "
        "Positive = expected surplus (can sell), negative = expected deficit (needs to buy)."
    )
    confidence: float = Field(ge=0, le=1, description="Model confidence, 0-1, for display/debugging only")


# ---------------------------------------------------------------------------
# Agents -> Simulation
# ---------------------------------------------------------------------------

class AgentDecision(BaseModel):
    """One household's order for a given simulation tick.

    Produced by: agents/
    Consumed by: simulation/ (market clearing)
    """

    household_id: str
    tick: int
    side: OrderSide
    quantity_kwh: float = Field(gt=0)
    limit_price_eur_per_kwh: float = Field(
        ge=0,
        description="For a SELL: minimum acceptable price. For a BUY: maximum acceptable price.",
    )
    strategy_name: str = Field(description="e.g. 'rule_based_baseline', 'ppo_v1' — for evaluation breakdowns")


# ---------------------------------------------------------------------------
# Simulation -> Dashboard
# ---------------------------------------------------------------------------

class TradeEvent(BaseModel):
    """A single cleared trade between two households in a given tick.

    Produced by: simulation/ (market clearing engine)
    Consumed by: dashboard/ (persistence + visualisation), agents/ (for next-tick learning signal)
    """

    trade_id: str
    tick: int
    timestamp: datetime
    buyer_id: str
    seller_id: str
    quantity_kwh: float = Field(gt=0)
    clearing_price_eur_per_kwh: float = Field(ge=0)


class MarketState(BaseModel):
    """Snapshot of market-wide state after clearing for a given tick.

    Produced by: simulation/
    Consumed by: dashboard/ (persistence + visualisation), agents/ (market context for next decision)
    """

    tick: int
    timestamp: datetime
    trades: list[TradeEvent]
    unmatched_buy_orders: list[AgentDecision]
    unmatched_sell_orders: list[AgentDecision]
    clearing_price_eur_per_kwh: float | None = Field(
        default=None, description="None if no trades cleared this tick"
    )


# ---------------------------------------------------------------------------
# Dashboard-level rollups (for the outcomes narrative, not per-tick logic)
# ---------------------------------------------------------------------------

class HouseholdOutcome(BaseModel):
    """Cumulative outcome for one household over a simulation run — this is
    what the dashboard's headline numbers are built from.
    """

    household_id: str
    total_cost_eur_p2p: float = Field(description="Actual cost/revenue under P2P trading")
    total_cost_eur_grid_baseline: float = Field(
        description="Hypothetical cost/revenue if the household had only used grid import/export tariffs"
    )
    savings_eur: float = Field(description="grid_baseline - p2p. Positive = P2P trading saved money.")
    savings_pct: float


class RunSummary(BaseModel):
    """Top-level result of a full simulation run — the number you defend in the demo."""

    run_id: str
    strategy_name: str = Field(description="Which agent strategy produced this run")
    num_households: int
    num_ticks: int
    total_savings_eur: float
    avg_savings_pct: float
    peak_load_reduction_pct: float
    co2_avoided_kg: float
