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


BASELINE_ACTION = np.zeros(1)  # the residual action space's origin is the baseline


def run_episode(env: GridPeerParallelEnv, policy, seed: int = 0) -> float:
    """Community savings in EUR over one episode, with ``policy(agent) -> action``.

    Summed from each household's own savings in ``infos``, not from rewards: a
    shared or mixed reward would count the community total more than once.
    """
    env.reset(seed=seed)
    total = 0.0
    while env.agents:
        actions = {agent: policy(agent) for agent in env.agents}
        _, _, _, _, infos = env.step(actions)
        total += sum(info["savings_eur"] for info in infos.values())
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
    env = GridPeerParallelEnv(community(), random_start=False)

    total = run_episode(env, lambda agent: BASELINE_ACTION)

    pipeline = run_pipeline(demo_households(), run_id="equivalence")
    assert total == pytest.approx(pipeline.summary.total_savings_eur, abs=1e-9)


def test_offering_nothing_saves_exactly_nothing():
    """Sitting every tick out leaves every household on grid tariffs: zero savings."""
    env = GridPeerParallelEnv(community(), random_start=False)

    total = run_episode(env, lambda agent: np.array([-1.0]))

    assert total == pytest.approx(0.0, abs=1e-12)


def test_episodes_are_reproducible_from_the_seed():
    env = GridPeerParallelEnv(community(), episode_ticks=48)
    rng = np.random.default_rng(3)
    fixed_actions = rng.uniform(-1, 1, size=(48, 4, 1))

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
            env.step({a: np.zeros(1) for a in env.agents})
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


def test_the_zero_action_is_the_baseline_order():
    for net_kwh in (0.3, 1.5, -0.7, -3.0):
        mine = decision_from_action([0.0], _forecast(net_kwh), _profile(), "rl")
        theirs = RuleBasedTrader().decide(_forecast(net_kwh), _profile())
        assert (mine.side, mine.quantity_kwh, mine.limit_price_eur_per_kwh) == (
            theirs.side,
            theirs.quantity_kwh,
            theirs.limit_price_eur_per_kwh,
        )


def test_action_scales_quantity_from_nothing_to_double():
    assert decision_from_action([-0.5], _forecast(2.0), _profile(), "rl").quantity_kwh == (
        pytest.approx(1.0)
    )
    assert decision_from_action([1.0], _forecast(2.0), _profile(), "rl").quantity_kwh == (
        pytest.approx(4.0)
    )
    assert decision_from_action([-1.0], _forecast(2.0), _profile(), "rl") is None


def test_offering_more_does_not_change_the_price():
    """Price comes from the forecast position, so extra volume never buys priority."""
    prices = {
        decision_from_action([a], _forecast(1.0), _profile(), "rl").limit_price_eur_per_kwh
        for a in (-0.5, 0.0, 0.5, 1.0)
    }
    assert len(prices) == 1


def test_side_follows_the_forecast_sign():
    assert decision_from_action([0.0], _forecast(1.0), _profile(), "rl").side is OrderSide.SELL
    assert decision_from_action([0.0], _forecast(-1.0), _profile(), "rl").side is OrderSide.BUY


def test_out_of_range_actions_are_clipped_not_rejected():
    decision = decision_from_action([7.0], _forecast(1.0), _profile(), "rl")
    assert decision.quantity_kwh == pytest.approx(2.0)


def test_observation_is_finite_and_the_documented_size():
    obs = build_observation(_forecast(1.5), _profile())
    assert obs.shape == (OBSERVATION_SIZE,)
    assert np.all(np.isfinite(obs))


# ---------------------------------------------------------------------------
# Reward modes
# ---------------------------------------------------------------------------


def _one_step(mode: str) -> tuple[dict, dict]:
    env = GridPeerParallelEnv(community(), random_start=False, reward_mode=mode)
    env.reset(seed=0)
    for _ in range(26):  # step into daylight, where trades happen
        env.step({a: BASELINE_ACTION for a in env.agents})
    _, rewards, _, _, infos = env.step({a: BASELINE_ACTION for a in env.agents})
    own = {a: info["savings_eur"] for a, info in infos.items()}
    return rewards, own


def test_individual_reward_is_each_households_own_savings():
    rewards, own = _one_step("individual")
    for agent, reward in rewards.items():
        assert reward == pytest.approx(own[agent] * REWARD_SCALE)


def test_community_reward_gives_everyone_the_community_total():
    rewards, own = _one_step("community")
    community_total = sum(own.values()) * REWARD_SCALE
    assert all(reward == pytest.approx(community_total) for reward in rewards.values())


def test_mixed_reward_is_half_own_half_community_share():
    rewards, own = _one_step("mixed")
    share = sum(own.values()) / len(own)
    for agent, reward in rewards.items():
        assert reward == pytest.approx((0.5 * own[agent] + 0.5 * share) * REWARD_SCALE)


def test_own_savings_sum_to_the_community_total_in_every_mode():
    totals = {mode: sum(_one_step(mode)[1].values()) for mode in ("mixed", "community")}
    assert totals["mixed"] == pytest.approx(totals["community"])


def test_unknown_reward_mode_is_rejected():
    with pytest.raises(ValueError, match="reward_mode"):
        GridPeerParallelEnv(community(), reward_mode="selfish")


# ---------------------------------------------------------------------------
# SB3 adapter
# ---------------------------------------------------------------------------


def test_vec_env_presents_one_slot_per_household_and_resets_together():
    env = GridPeerParallelEnv(community(), episode_ticks=5)
    vec = SharedPolicyVecEnv(env, seed=0)

    obs = vec.reset()
    assert obs.shape == (4, OBSERVATION_SIZE)

    for step in range(5):
        vec.step_async(np.zeros((4, 1), dtype=np.float32))
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
