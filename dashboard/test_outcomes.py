"""Tests for the run rollups — the arithmetic behind every number the demo quotes.

These pin the conventions (sign, weighting, zero-baseline handling) rather than
specific simulation results, because those conventions are what would silently
diverge between this module and agents/evaluation.py.
"""

import pytest

from dashboard.outcomes import (
    IE_GRID_CO2_G_PER_KWH,
    avg_savings_pct,
    co2_avoided_kg,
    compute_household_outcome,
    compute_household_outcomes,
    compute_run_summary,
    peak_load_reduction_pct,
)
from shared.schemas import HouseholdOutcome


def _outcome(household_id: str, p2p: float, baseline: float) -> HouseholdOutcome:
    return compute_household_outcome(household_id, p2p, baseline)


# -- HouseholdOutcome -------------------------------------------------------

def test_consumer_saving_money() -> None:
    """A buyer that paid less than the import tariff saved the difference."""
    outcome = _outcome("hh_000", p2p=45.0, baseline=60.0)
    assert outcome.savings_eur == pytest.approx(15.0)
    assert outcome.savings_pct == pytest.approx(25.0)


def test_producer_earning_more_reports_positive_pct() -> None:
    """A net seller's costs are negative (revenue); earning more must read positive.

    Earning EUR 30 instead of EUR 20 is a EUR 10 gain, measured against the
    magnitude of the baseline — not a sign-flipped -50%.
    """
    outcome = _outcome("hh_003", p2p=-30.0, baseline=-20.0)
    assert outcome.savings_eur == pytest.approx(10.0)
    assert outcome.savings_pct == pytest.approx(50.0)


def test_losing_money_reports_negative_savings() -> None:
    """The rollup reports a loss honestly rather than flooring at zero."""
    outcome = _outcome("hh_000", p2p=70.0, baseline=60.0)
    assert outcome.savings_eur == pytest.approx(-10.0)
    assert outcome.savings_pct == pytest.approx(-16.666667)


def test_zero_baseline_yields_zero_pct_not_a_division_error() -> None:
    outcome = _outcome("hh_000", p2p=0.0, baseline=0.0)
    assert outcome.savings_pct == 0.0


def test_outcomes_are_ordered_and_complete() -> None:
    outcomes = compute_household_outcomes(
        p2p_cost_eur={"hh_001": 10.0, "hh_000": 20.0},
        grid_baseline_cost_eur={"hh_000": 25.0, "hh_001": 12.0},
    )
    assert [o.household_id for o in outcomes] == ["hh_000", "hh_001"]
    assert outcomes[0].savings_eur == pytest.approx(5.0)


def test_mismatched_household_sets_raise() -> None:
    """A household settled on one side but not the other would bias the totals."""
    with pytest.raises(ValueError, match="different households"):
        compute_household_outcomes({"hh_000": 1.0}, {"hh_000": 1.0, "hh_001": 2.0})


# -- avg_savings_pct --------------------------------------------------------

def test_avg_savings_pct_is_baseline_weighted_not_a_plain_mean() -> None:
    """One tiny producer with a huge percentage must not dominate the run average.

    Plain mean here would be (1% + 100%) / 2 = 50.5%. Weighted by what each
    household actually had at stake it is 2.0 / 101.0 = ~2%, which is the figure
    that survives being questioned.
    """
    outcomes = [_outcome("big", p2p=99.0, baseline=100.0), _outcome("tiny", p2p=0.0, baseline=1.0)]
    assert outcomes[0].savings_pct == pytest.approx(1.0)
    assert outcomes[1].savings_pct == pytest.approx(100.0)
    assert avg_savings_pct(outcomes) == pytest.approx(2.0 / 101.0 * 100.0)


def test_avg_savings_pct_with_no_outcomes_is_zero() -> None:
    assert avg_savings_pct([]) == 0.0


# -- peak_load_reduction_pct ------------------------------------------------

def test_peak_load_reduction_compares_peaks_not_totals() -> None:
    """Peak against peak: the worst single tick is what strains a network."""
    assert peak_load_reduction_pct([1.0, 8.0, 1.0], [1.0, 10.0, 5.0]) == pytest.approx(20.0)


def test_peak_load_reduction_is_zero_when_the_peak_tick_cannot_trade() -> None:
    """A solar-only market does not help an evening peak — report 0%, not a fudge."""
    assert peak_load_reduction_pct([5.0, 2.0], [5.0, 4.0]) == pytest.approx(0.0)


def test_peak_load_reduction_with_no_grid_only_load_is_zero() -> None:
    assert peak_load_reduction_pct([0.0], [0.0]) == 0.0
    assert peak_load_reduction_pct([], []) == 0.0


def test_peak_load_reduction_can_be_negative() -> None:
    """If P2P somehow drew more at peak, say so rather than clamping."""
    assert peak_load_reduction_pct([12.0], [10.0]) == pytest.approx(-20.0)


# -- co2_avoided_kg ---------------------------------------------------------

def test_co2_avoided_converts_grams_to_kg() -> None:
    assert co2_avoided_kg(10.0, grid_co2_g_per_kwh=300.0) == pytest.approx(3.0)


def test_co2_avoided_uses_the_documented_default_factor() -> None:
    assert IE_GRID_CO2_G_PER_KWH == 300.0
    assert co2_avoided_kg(1.0) == pytest.approx(0.3)


def test_co2_avoided_rejects_negative_volume() -> None:
    with pytest.raises(ValueError):
        co2_avoided_kg(-1.0)


# -- RunSummary -------------------------------------------------------------

def test_run_summary_assembles_every_headline_number() -> None:
    outcomes = [
        _outcome("hh_000", p2p=45.0, baseline=60.0),
        _outcome("hh_001", p2p=-30.0, baseline=-20.0),
    ]
    summary = compute_run_summary(
        run_id="run_a",
        strategy_name="rule_based_baseline",
        outcomes=outcomes,
        num_ticks=48,
        traded_kwh=100.0,
        p2p_grid_import_kwh_per_tick=[8.0, 1.0],
        grid_only_import_kwh_per_tick=[10.0, 1.0],
    )
    assert summary.run_id == "run_a"
    assert summary.num_households == 2
    assert summary.num_ticks == 48
    assert summary.total_savings_eur == pytest.approx(25.0)
    assert summary.avg_savings_pct == pytest.approx(25.0 / 80.0 * 100.0)
    assert summary.peak_load_reduction_pct == pytest.approx(20.0)
    assert summary.co2_avoided_kg == pytest.approx(30.0)
