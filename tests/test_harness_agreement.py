"""The agents evaluation harness and the dashboard pipeline must report one answer.

There are two implementations of the run rollup: ``agents.evaluation.run_strategy``
(the harness the RL comparison is built on) and ``dashboard.orchestrator.run_pipeline``
(what the dashboard persists and renders). Each lives in its own module so neither has
to import the other. Until the team settles on a single home for the arithmetic, this
test is the guarantee that they cannot drift: the project's headline number must not
depend on which command produced it.

Both are run on identical households with the same strategy and must agree on every
RunSummary field and every household's outcome.
"""

import pytest

from agents.baseline import STRATEGY_NAME, RuleBasedTrader
from agents.evaluation import HouseholdSeries as HarnessSeries
from agents.evaluation import run_strategy
from dashboard.orchestrator import HouseholdSeries as PipelineSeries
from dashboard.orchestrator import run_pipeline
from data.synthetic import default_profiles, generate_household_series

SUMMARY_FIELDS = (
    "num_households",
    "num_ticks",
    "total_savings_eur",
    "avg_savings_pct",
    "peak_load_reduction_pct",
    "co2_avoided_kg",
)


def _cohort(num_households: int, days: int, seed: int) -> list[PipelineSeries]:
    """A synthetic cohort in the dashboard's series type."""
    cohort = []
    for profile in default_profiles(num_households):
        demand, solar = generate_household_series(profile, days=days, seed=seed)
        cohort.append(PipelineSeries(profile=profile, demand_kwh=demand, solar_kwh=solar))
    return cohort


@pytest.mark.parametrize(
    ("num_households", "days", "seed"),
    [(4, 2, 0), (8, 3, 1), (12, 2, 7)],
)
def test_harness_and_pipeline_report_identical_results(num_households, days, seed):
    cohort = _cohort(num_households, days, seed)

    harness = run_strategy(
        [HarnessSeries(h.profile, h.demand_kwh, h.solar_kwh) for h in cohort],
        RuleBasedTrader(),
        run_id="agreement",
        strategy_name=STRATEGY_NAME,
    )
    pipeline = run_pipeline(cohort, run_id="agreement", strategy_name=STRATEGY_NAME)

    for field in SUMMARY_FIELDS:
        assert getattr(harness.summary, field) == pytest.approx(
            getattr(pipeline.summary, field), abs=1e-9
        ), f"harness and dashboard disagree on {field}"

    harness_by_id = {o.household_id: o for o in harness.outcomes}
    for outcome in pipeline.outcomes:
        mine = harness_by_id[outcome.household_id]
        assert mine.savings_eur == pytest.approx(outcome.savings_eur, abs=1e-9)
        assert mine.savings_pct == pytest.approx(outcome.savings_pct, abs=1e-9)
