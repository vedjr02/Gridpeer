"""The dashboard pipeline and the Stage 5 experiment are one system, not two.

With batteries on, ``dashboard.run_pipeline`` now steps the battery inside the loop,
hands each strategy the battery state (schema 0.2.0) and measures savings against the
status quo. ``agents.battery_experiment.rollout`` does the same through
``MarketSimulator``. If they ever disagree, a number shown on the dashboard would not
be the number the experiment reported — so they are pinned together here.
"""

import pytest

from agents.baseline import RuleBasedTrader
from agents.battery_experiment import MAX_BATTERY_POWER_KW, baseline_decider, rollout
from agents.train_rl import cohort
from dashboard.orchestrator import HouseholdSeries as PipelineSeries
from dashboard.orchestrator import run_pipeline
from shared.schemas import HouseholdState


def _pipeline_series(seed: int) -> list[PipelineSeries]:
    return [PipelineSeries(h.profile, h.demand_kwh, h.solar_kwh) for h in cohort(4, 2, seed)]


@pytest.mark.parametrize("seed", [5000, 5001, 7003])
def test_battery_aware_baseline_matches_the_experiment_exactly(seed):
    pipeline = run_pipeline(
        _pipeline_series(seed),
        run_id="pinned",
        use_battery=True,
        battery_power_kw=MAX_BATTERY_POWER_KW,
    )
    experiment = rollout(cohort(4, 2, seed), baseline_decider(aware=True))

    assert pipeline.summary.total_savings_eur == pytest.approx(experiment.savings_eur, abs=1e-9)
    by_household = {o.household_id: o.savings_eur for o in pipeline.outcomes}
    for household_id, saved in experiment.household_savings_eur.items():
        assert by_household[household_id] == pytest.approx(saved, abs=1e-9)


def test_strategies_receive_the_battery_state_when_batteries_are_on():
    seen: list[HouseholdState | None] = []

    class Spy(RuleBasedTrader):
        def decide(self, forecast, profile, state=None):
            seen.append(state)
            return super().decide(forecast, profile, state)

    run_pipeline(_pipeline_series(5000), num_ticks=6, run_id="spy", strategy=Spy(),
                 use_battery=True, battery_power_kw=2.5)

    assert seen and all(isinstance(state, HouseholdState) for state in seen)
    assert {state.tick for state in seen} == set(range(6))
    assert all(state.battery_max_power_kw == 2.5 for state in seen)


def test_a_battery_offset_on_an_order_reaches_the_battery():
    """An order that asks to discharge more really does move the battery."""
    class DischargeEverything(RuleBasedTrader):
        def decide(self, forecast, profile, state=None):
            order = super().decide(forecast, profile, state)
            if order is None or state is None or state.battery_capacity_kwh == 0:
                return order
            return order.model_copy(update={"battery_offset_kwh": 5.0})

    households = _pipeline_series(5000)
    automatic = run_pipeline(households, run_id="auto", use_battery=True, battery_power_kw=2.5)
    forced = run_pipeline(households, run_id="forced", strategy=DischargeEverything(),
                          use_battery=True, battery_power_kw=2.5)

    assert forced.summary.total_savings_eur != pytest.approx(automatic.summary.total_savings_eur)
    assert any(d.battery_offset_kwh == 5.0 for d in forced.decisions)
