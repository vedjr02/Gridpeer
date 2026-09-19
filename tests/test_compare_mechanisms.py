"""Smoke tests for the pricing-rule comparison, and the finding it rests on."""

import pytest

from scripts.compare_mechanisms import compare
from simulation.market import PricingRule


@pytest.fixture(scope="module")
def outcomes():
    """Both rules on one small, identical community over two days."""
    return compare(num_households=8, days=2)


def test_both_rules_trade_exactly_the_same_energy(outcomes):
    """Matching does not depend on the pricing rule, so neither does volume."""
    pairwise = outcomes[PricingRule.PAIRWISE_MIDPOINT]
    uniform = outcomes[PricingRule.UNIFORM]

    assert pairwise.traded_kwh > 0
    assert uniform.traded_kwh == pytest.approx(pairwise.traded_kwh)


def test_the_pricing_rule_cannot_change_community_savings(outcomes):
    """What one neighbour pays, another receives: price only moves savings between them.

    With the same trades, the community's total saving is fixed by the volume that
    avoided the grid, whatever each kWh changed hands at. The rule can change *who*
    saves — or change behaviour, once strategies respond to it — but not the total.
    """
    pairwise = outcomes[PricingRule.PAIRWISE_MIDPOINT]
    uniform = outcomes[PricingRule.UNIFORM]

    assert uniform.savings_eur == pytest.approx(pairwise.savings_eur)


def test_uniform_pricing_charges_every_neighbour_the_same_in_a_tick(outcomes):
    assert outcomes[PricingRule.UNIFORM].mean_price_spread_eur_per_kwh == pytest.approx(0.0)
