"""Smoke tests for the Stage 6 realism study's building blocks."""

import pytest

from agents.battery_experiment import baseline_decider
from agents.train_rl import cohort
from scripts.realism_study import Strategy, measure, scaled_solar


def test_scaled_solar_scales_only_solar():
    households = cohort(4, 1, seed=0)
    winter = scaled_solar(households, 0.2)
    for original, scaled in zip(households, winter, strict=True):
        assert scaled.demand_kwh == original.demand_kwh
        assert scaled.solar_kwh == pytest.approx([kwh * 0.2 for kwh in original.solar_kwh])


def test_less_sun_means_less_saving_for_the_baseline():
    baseline = Strategy("baseline", baseline_decider(aware=False), battery=False)
    summer = measure(baseline, 4, cohorts=2)
    winter = measure(baseline, 4, cohorts=2, transform=lambda h: scaled_solar(h, 0.2))
    assert winter.savings_eur_per_household_day < summer.savings_eur_per_household_day


def test_a_network_charge_costs_the_trading_baseline_money():
    baseline = Strategy("baseline", baseline_decider(aware=False), battery=False)
    free = measure(baseline, 4, cohorts=2)
    charged = measure(baseline, 4, cohorts=2, network_charge_eur_per_kwh=0.05)
    assert charged.savings_eur_per_household_day < free.savings_eur_per_household_day
