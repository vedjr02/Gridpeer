"""A trained RL policy, served through the same ``decide()`` as the baseline.

The pipeline asks every strategy one question — ``decide(forecast, profile)`` — so a
trained policy plugs in exactly where ``RuleBasedTrader`` does, with no change to the
loop, the persistence or the dashboard. Observation and action encoding come from
``agents.rl_env``, the same functions the policy was trained with.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

from agents.rl_env import build_observation, decision_from_action
from shared.schemas import AgentDecision, ForecastOutput, HouseholdProfile

DEFAULT_MODEL_PATH = Path("models/ppo_v1.zip")
DEFAULT_STRATEGY_NAME = "ppo_v1"


class PPOTrader:
    """One shared PPO policy deciding for every household, deterministically.

    Deterministic (the policy's mean action, no sampling) so that a run can be
    replayed exactly — the same property the baseline has and the evaluation
    harness relies on.
    """

    def __init__(
        self,
        model: PPO | None = None,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        strategy_name: str = DEFAULT_STRATEGY_NAME,
    ) -> None:
        """model: an in-memory policy; otherwise one is loaded from ``model_path``."""
        if model is None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"no trained policy at {path}; train one with `python -m agents.train_rl`"
                )
            model = PPO.load(path, device="cpu")
        self.model = model
        self.strategy_name = strategy_name

    def decide(
        self, forecast: ForecastOutput, profile: HouseholdProfile
    ) -> AgentDecision | None:
        """This household's order for the tick, or None if the policy offers nothing."""
        observation = build_observation(forecast, profile)
        action, _ = self.model.predict(observation, deterministic=True)
        return decision_from_action(action, forecast, profile, self.strategy_name)
