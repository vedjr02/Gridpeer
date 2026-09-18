"""Tests for battery control in the RL environment and the Stage 5 experiment."""

import numpy as np
import pytest

from agents.battery_experiment import (
    MAX_BATTERY_POWER_KW,
    Outcome,
    baseline_decider,
    rollout,
    without_batteries,
)
from agents.rl_env import (
    BATTERY_ACTION_SIZE,
    BATTERY_OBSERVATION_SIZE,
    GridPeerParallelEnv,
    battery_decision,
    status_quo_cost_eur,
)
from agents.train_rl import cohort
from shared.schemas import OrderSide


def battery_env(**kwargs) -> GridPeerParallelEnv:
    return GridPeerParallelEnv(
        cohort(4, 2, seed=11),
        random_start=False,
        use_battery=True,
        battery_control=True,
        max_battery_power_kw=MAX_BATTERY_POWER_KW,
        **kwargs,
    )


def episode_savings(env: GridPeerParallelEnv, action) -> tuple[float, float]:
    """(real savings from infos, total training reward) over one episode."""
    env.reset(seed=0)
    savings = reward = 0.0
    while env.agents:
        _, rewards, _, _, infos = env.step({a: np.asarray(action) for a in env.agents})
        savings += sum(i["savings_eur"] for i in infos.values())
        reward += sum(rewards.values())
    return savings, reward


def test_battery_control_needs_batteries():
    with pytest.raises(ValueError, match="use_battery"):
        GridPeerParallelEnv(cohort(4, 2, seed=0), battery_control=True)


def test_battery_mode_widens_observation_and_action():
    env = battery_env()
    agent = env.possible_agents[0]
    assert env.observation_space(agent).shape == (BATTERY_OBSERVATION_SIZE,)
    assert env.action_space(agent).shape == (BATTERY_ACTION_SIZE,)


def test_zero_action_is_the_battery_aware_baseline_exactly():
    """Through the environment, doing nothing reproduces the strong heuristic.

    The RL policy starts here, so anything it learns is measured against the
    rule-based baseline trading with full knowledge of the automatic battery.
    """
    households = cohort(4, 2, seed=11)
    savings, _ = episode_savings(battery_env(), np.zeros(2))

    heuristic = rollout(households, baseline_decider(aware=True))

    assert savings == pytest.approx(heuristic.savings_eur, abs=1e-9)


def test_shaping_changes_the_training_signal_but_not_the_reported_savings():
    shaped_savings, shaped_reward = episode_savings(battery_env(shaping_gamma=0.99), [0.0, 0.3])
    plain_savings, plain_reward = episode_savings(battery_env(), [0.0, 0.3])

    assert shaped_savings == pytest.approx(plain_savings)
    assert shaped_reward != pytest.approx(plain_reward)


def test_status_quo_cost_is_grid_tariffs_on_the_raw_position():
    profile = cohort(4, 1, seed=0)[0].profile
    assert status_quo_cost_eur(profile, 2.0) == pytest.approx(
        -2.0 * profile.grid_export_tariff_eur_per_kwh
    )
    assert status_quo_cost_eur(profile, -1.5) == pytest.approx(
        1.5 * profile.grid_import_tariff_eur_per_kwh
    )


def test_positive_battery_action_turns_stored_energy_into_a_sell_order():
    """Evening, battery charged: discharging extra is planned in, so it is offered."""
    env = battery_env()
    env.reset(seed=0)
    producer = "hh_003"
    while env.current_forecast(producer).timestamp.hour < 19:
        env.step({a: np.zeros(2) for a in env.agents})
    environment = env._simulator.environments[producer]
    assert environment.battery_level_kwh > 1.0, "the producer should have stored midday solar"

    order, offset = battery_decision(
        [0.0, 1.0], env.current_forecast(producer), env._profiles[producer], environment, "t"
    )
    assert offset == pytest.approx(environment.max_flow_kwh)
    assert order is not None and order.side is OrderSide.SELL


def test_rollout_reports_zero_savings_for_the_status_quo_itself():
    """No battery, and nobody trades: the bill is exactly the status quo."""
    households = without_batteries(cohort(4, 2, seed=3))
    outcome = rollout(households, lambda simulator, forecasts: ([], None))
    assert outcome.savings_eur == pytest.approx(0.0, abs=1e-12)
    assert outcome.peak_reduction_pct == pytest.approx(0.0)


def test_peak_reduction_is_relative_to_the_status_quo_peak():
    outcome = Outcome(peak_import_kwh=1.5, status_quo_peak_kwh=2.0)
    assert outcome.peak_reduction_pct == pytest.approx(25.0)
