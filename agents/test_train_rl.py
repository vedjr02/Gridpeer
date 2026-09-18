"""Tests for the training script's selection rule and the deployed policy wrapper."""

import pytest

from agents.rl_env import GridPeerParallelEnv
from agents.rl_policy import PPOTrader
from agents.rl_vec_env import SharedPolicyVecEnv
from agents.train_rl import Comparison, cohort, select
from shared.schemas import AgentDecision


def _comparison(uplift_eur: float, worst_change_eur: float) -> Comparison:
    """A comparison with a given community gain and worst household change."""
    return Comparison(
        baseline_eur=[10.0],
        policy_eur=[10.0 + uplift_eur],
        household_change_eur={"hh_a": worst_change_eur, "hh_b": 1.0},
    )


def test_a_fair_seed_beats_a_bigger_but_unfair_one():
    """Fairness first: more community savings never excuses a household losing out."""
    unfair_but_big = (0, "model0", _comparison(uplift_eur=5.0, worst_change_eur=-0.2))
    fair_but_smaller = (1, "model1", _comparison(uplift_eur=1.0, worst_change_eur=0.1))

    assert select([unfair_but_big, fair_but_smaller])[0] == 1


def test_among_fair_seeds_the_biggest_uplift_wins():
    candidates = [
        (0, "m0", _comparison(uplift_eur=1.0, worst_change_eur=0.0)),
        (1, "m1", _comparison(uplift_eur=3.0, worst_change_eur=0.2)),
    ]
    assert select(candidates)[0] == 1


def test_with_no_fair_seed_the_least_unfair_wins():
    candidates = [
        (0, "m0", _comparison(uplift_eur=5.0, worst_change_eur=-0.5)),
        (1, "m1", _comparison(uplift_eur=1.0, worst_change_eur=-0.1)),
    ]
    assert select(candidates)[0] == 1


def test_comparison_reports_uplift_wins_and_fairness():
    result = Comparison(
        baseline_eur=[1.0, 1.0],
        policy_eur=[1.5, 0.9],
        household_change_eur={"a": 0.3, "b": 0.0},
    )
    assert result.uplift_pct == pytest.approx(20.0)
    assert result.wins == 1
    assert result.fair


def test_ppo_trader_serves_decisions_through_decide():
    """A trained policy plugs in wherever the baseline does, deterministically."""
    from stable_baselines3 import PPO

    env = GridPeerParallelEnv(cohort(4, 2, seed=0), episode_ticks=48)
    model = PPO("MlpPolicy", SharedPolicyVecEnv(env, seed=0), n_steps=16, batch_size=64,
                n_epochs=1, seed=0, device="cpu")
    trader = PPOTrader(model=model)

    env.reset(seed=0)
    for _ in range(26):
        env.step({a: [0.0] for a in env.agents})
    forecast = env.current_forecast("hh_000")
    profile = env._profiles["hh_000"]

    first = trader.decide(forecast, profile)
    second = trader.decide(forecast, profile)
    assert first is None or isinstance(first, AgentDecision)
    assert first == second
    if first is not None:
        assert first.strategy_name == "ppo_v1"


def test_missing_model_file_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_rl"):
        PPOTrader(model_path=tmp_path / "nope.zip")
