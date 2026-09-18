"""Tests for the RL environment and its SB3 adapter."""

from datetime import datetime

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from agents.baseline import RuleBasedTrader
from agents.rl_env import (
    OBSERVATION_SIZE,
    REWARD_SCALE,
    GridPeerParallelEnv,
    build_observation,
    decision_from_action,
)
from agents.rl_vec_env import SharedPolicyVecEnv
from dashboard.orchestrator import demo_households, run_pipeline
from shared.schemas import AgentRole, ForecastOutput, HouseholdProfile, OrderSide
from simulation.simulator import HouseholdSeries


def community() -> list[HouseholdSeries]:
    """The dashboard's demo households, in the simulator's series type."""
    return [
        HouseholdSeries(profile=h.profile, demand_kwh=h.demand_kwh, solar_kwh=h.solar_kwh)
        for h in demo_households()
    ]


def baseline_action(trader: RuleBasedTrader, forecast: ForecastOutput) -> np.ndarray:
    """The action that reproduces the rule-based baseline's order exactly."""
    eagerness = trader._eagerness(abs(forecast.predicted_net_position_kwh))
    return np.array([1.0, 2.0 * eagerness - 1.0])


def run_episode(env: GridPeerParallelEnv, policy, seed: int = 0) -> float:
    """Total savings in EUR over one episode, with ``policy(agent) -> action``."""
    env.reset(seed=seed)
    total = 0.0
    while env.agents:
        actions = {agent: policy(agent) for agent in env.agents}
        _, rewards, _, _, _ = env.step(actions)
        total += sum(rewards.values()) / REWARD_SCALE
    return total


def test_follows_the_pettingzoo_parallel_api():
    env = GridPeerParallelEnv(community(), episode_ticks=48)
    parallel_api_test(env, num_cycles=60)


def test_baseline_actions_reproduce_the_dashboard_pipeline_exactly():
    """The environment is the dashboard's market, not an approximation of it.

    Driven with the baseline's decisions over the full series from a cold start, it
    must land on run_pipeline's total savings. If it did not, a policy would be
    trained on one market and judged on another.
    """
    households = community()
    trader = RuleBasedTrader()
    env = GridPeerParallelEnv(households, random_start=False)

    total = run_episode(env, lambda agent: baseline_action(trader, env.current_forecast(agent)))

    pipeline = run_pipeline(demo_households(), run_id="equivalence")
    assert total == pytest.approx(pipeline.summary.total_savings_eur, abs=1e-9)


def test_offering_nothing_saves_exactly_nothing():
    """Sitting every tick out leaves every household on grid tariffs: zero savings."""
    env = GridPeerParallelEnv(community(), random_start=False)

    total = run_episode(env, lambda agent: np.array([-1.0, 0.0]))

    assert total == pytest.approx(0.0, abs=1e-12)


def test_episodes_are_reproducible_from_the_seed():
    env = GridPeerParallelEnv(community(), episode_ticks=48)
    rng = np.random.default_rng(3)
    fixed_actions = rng.uniform(-1, 1, size=(48, 4, 2))

    def replay(seed: int) -> list[float]:
        env.reset(seed=seed)
        rewards_seen, t = [], 0
        while env.agents:
            actions = {a: fixed_actions[t, i] for i, a in enumerate(env.possible_agents)}
            _, rewards, _, _, _ = env.step(actions)
            rewards_seen.append(sum(rewards.values()))
            t += 1
        return rewards_seen

    assert replay(11) == replay(11)


def test_episode_length_and_random_start_stay_in_bounds():
    env = GridPeerParallelEnv(community(), episode_ticks=10)
    for seed in range(20):
        env.reset(seed=seed)
        steps = 0
        while env.agents:
            env.step({a: np.zeros(2) for a in env.agents})
            steps += 1
        assert steps == 10


def test_invalid_episode_length_is_rejected():
    with pytest.raises(ValueError, match="episode_ticks"):
        GridPeerParallelEnv(community(), episode_ticks=10_000)


# ---------------------------------------------------------------------------
# Action and observation encoding
# ---------------------------------------------------------------------------


def _profile() -> HouseholdProfile:
    return HouseholdProfile(
        household_id="hh",
        role=AgentRole.PROSUMER,
        has_solar=True,
        solar_capacity_kw=3.0,
        battery_capacity_kwh=0.0,
        grid_export_tariff_eur_per_kwh=0.07,
        grid_import_tariff_eur_per_kwh=0.25,
    )


def _forecast(net_kwh: float) -> ForecastOutput:
    return ForecastOutput(
        household_id="hh",
        tick=5,
        timestamp=datetime(2026, 1, 1, 12, 0),
        predicted_demand_kwh=1.0,
        predicted_solar_generation_kwh=max(0.0, 1.0 + net_kwh),
        predicted_net_position_kwh=net_kwh,
        confidence=0.5,
    )


def test_action_maps_to_share_side_and_a_price_inside_the_gap():
    sell = decision_from_action([0.0, 0.0], _forecast(2.0), _profile(), "rl")
    assert sell.side is OrderSide.SELL
    assert sell.quantity_kwh == pytest.approx(1.0)  # 50% of 2.0 kWh
    assert sell.limit_price_eur_per_kwh == pytest.approx(0.07 + 0.5 * 0.18)

    buy = decision_from_action([1.0, 1.0], _forecast(-1.0), _profile(), "rl")
    assert buy.side is OrderSide.BUY
    assert buy.quantity_kwh == pytest.approx(1.0)
    assert buy.limit_price_eur_per_kwh == pytest.approx(0.25)  # fully eager buyer


def test_out_of_range_actions_are_clipped_not_rejected():
    decision = decision_from_action([7.0, -9.0], _forecast(1.0), _profile(), "rl")
    assert decision.quantity_kwh == pytest.approx(1.0)
    assert 0.07 <= decision.limit_price_eur_per_kwh <= 0.25


def test_a_zero_share_is_no_order_at_all():
    assert decision_from_action([-1.0, 0.0], _forecast(1.0), _profile(), "rl") is None


def test_observation_is_finite_and_the_documented_size():
    obs = build_observation(_forecast(1.5), _profile())
    assert obs.shape == (OBSERVATION_SIZE,)
    assert np.all(np.isfinite(obs))


# ---------------------------------------------------------------------------
# SB3 adapter
# ---------------------------------------------------------------------------


def test_vec_env_presents_one_slot_per_household_and_resets_together():
    env = GridPeerParallelEnv(community(), episode_ticks=5)
    vec = SharedPolicyVecEnv(env, seed=0)

    obs = vec.reset()
    assert obs.shape == (4, OBSERVATION_SIZE)

    for step in range(5):
        vec.step_async(np.zeros((4, 2), dtype=np.float32))
        obs, rewards, dones, infos = vec.step_wait()
        assert rewards.shape == (4,)
        assert dones.all() == (step == 4)

    assert all("terminal_observation" in info for info in infos)
    assert obs.shape == (4, OBSERVATION_SIZE)  # already the next episode's first obs


def test_ppo_trains_on_the_vec_env():
    """Smoke test: SB3's PPO accepts the adapter and completes a short run."""
    from stable_baselines3 import PPO

    vec = SharedPolicyVecEnv(GridPeerParallelEnv(community(), episode_ticks=48), seed=0)
    model = PPO("MlpPolicy", vec, n_steps=32, batch_size=64, n_epochs=1, seed=0, device="cpu")
    model.learn(total_timesteps=256)
