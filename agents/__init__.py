"""agents package: trading strategies (baseline now, RL later).

Consumes ForecastOutput and MarketState, produces AgentDecision — see shared/schemas.py.
"""

from agents.baseline import STRATEGY_NAME, RuleBasedTrader
from agents.evaluation import RunResult, TradingStrategy, compare, run_strategy

__all__ = [
    "STRATEGY_NAME",
    "RuleBasedTrader",
    "RunResult",
    "TradingStrategy",
    "compare",
    "run_strategy",
]
