"""simulation package: market clearing mechanism + household environment.

Consumes AgentDecision, produces TradeEvent and MarketState — see shared/schemas.py.
Settlement figures (P2P versus grid-only) are exposed per tick for the dashboard to
roll up into HouseholdOutcome and RunSummary.
"""

from simulation.environment import (
    HouseholdEnvironment,
    HouseholdState,
    battery_flow_kwh,
    max_flow_kwh_per_tick,
)
from simulation.market import PricingRule, clear_tick, trade_id
from simulation.settlement import HouseholdSettlement, settle_tick

__all__ = [
    "HouseholdEnvironment",
    "HouseholdSettlement",
    "HouseholdState",
    "PricingRule",
    "battery_flow_kwh",
    "max_flow_kwh_per_tick",
    "clear_tick",
    "settle_tick",
    "trade_id",
]
