"""Tests for the evaluation harness — the end-to-end run behind the headline number."""

import pytest

from agents.baseline import STRATEGY_NAME, RuleBasedTrader
from agents.evaluation import HouseholdSeries, compare, run_strategy
from shared.schemas import AgentRole, HouseholdProfile, RunSummary


def profile(
    household_id: str,
    role: AgentRole = AgentRole.PROSUMER,
    battery_capacity_kwh: float = 0.0,
) -> HouseholdProfile:
    """Battery defaults to zero here so the harness's trading behaviour is visible."""
    return HouseholdProfile(
        household_id=household_id,
        role=role,
        has_solar=role is not AgentRole.PURE_CONSUMER,
        solar_capacity_kw=3.0 if role is not AgentRole.PURE_CONSUMER else 0.0,
        battery_capacity_kwh=battery_capacity_kwh,
        grid_export_tariff_eur_per_kwh=0.07,
        grid_import_tariff_eur_per_kwh=0.25,
    )


def complementary_households(ticks: int = 8) -> list[HouseholdSeries]:
    """A steady producer and a steady consumer: the case P2P trading is built for."""
    producer = HouseholdSeries(
        profile=profile("hh_producer", AgentRole.PURE_PRODUCER),
        demand_kwh=[0.2] * ticks,
        solar_kwh=[2.2] * ticks,
    )
    consumer = HouseholdSeries(
        profile=profile("hh_consumer", AgentRole.PURE_CONSUMER),
        demand_kwh=[2.0] * ticks,
        solar_kwh=[0.0] * ticks,
    )
    return [producer, consumer]


def run_baseline(households: list[HouseholdSeries], **kwargs) -> "RunSummary":
    result = run_strategy(
        households,
        RuleBasedTrader(),
        run_id="test_run",
        strategy_name=STRATEGY_NAME,
        **kwargs,
    )
    return result


def test_a_complementary_pair_saves_both_households_money():
    """The project's thesis, end to end: surplus next door beats the grid for both."""
    result = run_baseline(complementary_households())

    assert result.summary.total_savings_eur > 0
    assert result.summary.avg_savings_pct > 0
    assert len(result.outcomes) == 2
    assert all(outcome.savings_eur > 0 for outcome in result.outcomes)


def test_trading_reduces_peak_grid_import():
    """Energy sourced from a neighbour is energy the grid did not have to carry."""
    result = run_baseline(complementary_households())

    assert result.summary.peak_load_reduction_pct > 0
    assert result.summary.co2_avoided_kg > 0


def test_households_that_never_complement_each_other_save_nothing():
    """Two consumers cannot trade: everything falls back to the grid, savings are zero.

    This is the honest floor of the measurement. If a run with no possible trade
    ever reported a saving, the headline number would be meaningless.
    """
    ticks = 6
    households = [
        HouseholdSeries(
            profile=profile(f"hh_consumer_{i}", AgentRole.PURE_CONSUMER),
            demand_kwh=[1.5] * ticks,
            solar_kwh=[0.0] * ticks,
        )
        for i in range(2)
    ]

    result = run_baseline(households)

    assert result.summary.total_savings_eur == pytest.approx(0.0)
    assert result.summary.co2_avoided_kg == pytest.approx(0.0)
    assert result.summary.peak_load_reduction_pct == pytest.approx(0.0)


def test_the_summary_satisfies_the_shared_contract():
    result = run_baseline(complementary_households())

    assert RunSummary.model_validate(result.summary.model_dump()) == result.summary
    assert result.summary.strategy_name == STRATEGY_NAME
    assert result.summary.num_households == 2
    assert result.summary.num_ticks == 8
    assert len(result.market_states) == 8


def test_runs_are_reproducible():
    """Same households, same strategy, same numbers — twice.

    The RL work depends on this: a policy comparison is worthless if the harness
    itself is noisy between runs.
    """
    first = run_baseline(complementary_households())
    second = run_baseline(complementary_households())

    assert first.summary.model_dump() == second.summary.model_dump()
    assert [o.model_dump() for o in first.outcomes] == [o.model_dump() for o in second.outcomes]


def test_num_ticks_caps_the_run():
    result = run_baseline(complementary_households(ticks=20), num_ticks=5)

    assert result.summary.num_ticks == 5
    assert len(result.market_states) == 5


def test_running_with_no_households_is_an_error():
    with pytest.raises(ValueError, match="no households"):
        run_strategy([], RuleBasedTrader(), run_id="r", strategy_name="s")


def test_compare_reports_a_loss_as_plainly_as_a_win():
    """A learned policy losing to the baseline is a legitimate result to report."""
    baseline = RunSummary(
        run_id="r1",
        strategy_name="rule_based_baseline",
        num_households=2,
        num_ticks=8,
        total_savings_eur=10.0,
        avg_savings_pct=20.0,
        peak_load_reduction_pct=5.0,
        co2_avoided_kg=3.0,
    )
    worse = baseline.model_copy(
        update={"strategy_name": "ppo_v1", "total_savings_eur": 7.5}
    )

    line = compare(baseline, worse)

    assert "2.50 less" in line
    assert "ppo_v1" in line
