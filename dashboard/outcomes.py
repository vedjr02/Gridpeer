"""Run rollups: per-tick settlement figures summed into the numbers the demo quotes.

dashboard/AGENTS.md names ``HouseholdOutcome`` and ``RunSummary`` as this module's
outputs, and the narrative it has to support is one sentence: *households using this
marketplace saved X% versus just exporting to the grid, and the learned agents beat
the rule-based ones by Y%.* Everything here exists to make that sentence defensible.

The arithmetic is deliberately dumb — sums and ratios over figures simulation/
already produced per tick. It does not re-derive what a tick cost; ``settle_tick``
owns that, and duplicating it here is how two versions of the headline number start
to disagree.

Sign convention, inherited from simulation/settlement.py: **cost is positive when
money leaves the household.** A net producer therefore has a negative cost, which is
revenue, and its savings percentage is measured against the magnitude of its
grid-only figure so that "earned more" reads as a positive percentage rather than a
sign-flipped one.

NOTE for the team: ``agents/evaluation.py`` currently computes its own copy of these
rollups and flags in its own docstring that the arithmetic belongs here. The two are
intended to agree; ``test_outcomes.py`` pins the shared conventions so a divergence
shows up as a failing test rather than as two different numbers in a slide deck.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from shared.schemas import HouseholdOutcome, RunSummary

# Ireland's grid carbon intensity, grams of CO2 per kWh of electricity consumed.
# 300 g/kWh is a round working figure in the range SEAI has published for recent
# years (Energy in Ireland, annual). It is an assumption, not a citation, and it
# scales the headline CO2 number linearly — so it is stated here, once, rather than
# buried at a call site, and the final writeup should replace it with a sourced
# figure for the specific year the demand data covers.
IE_GRID_CO2_G_PER_KWH = 300.0


def compute_household_outcome(
    household_id: str,
    total_cost_eur_p2p: float,
    total_cost_eur_grid_baseline: float,
) -> HouseholdOutcome:
    """Roll one household's run-long costs into its HouseholdOutcome contract.

    ``savings_pct`` is measured against the magnitude of the grid-only baseline, so
    a household that only ever earned an export tariff reports a positive
    percentage for earning more. A zero baseline yields 0.0 rather than a division
    by zero: a household with nothing at stake saved nothing.
    """
    savings_eur = total_cost_eur_grid_baseline - total_cost_eur_p2p
    reference_eur = abs(total_cost_eur_grid_baseline)
    return HouseholdOutcome(
        household_id=household_id,
        total_cost_eur_p2p=total_cost_eur_p2p,
        total_cost_eur_grid_baseline=total_cost_eur_grid_baseline,
        savings_eur=savings_eur,
        savings_pct=(savings_eur / reference_eur * 100.0) if reference_eur > 0 else 0.0,
    )


def compute_household_outcomes(
    p2p_cost_eur: Mapping[str, float],
    grid_baseline_cost_eur: Mapping[str, float],
) -> list[HouseholdOutcome]:
    """Roll every household's run-long costs up, ordered by household_id.

    Both mappings are household_id to EUR summed across the run. A household present
    in one but not the other is an error — it means a tick was settled for a
    household the run does not know about, which would quietly bias the totals.
    """
    if set(p2p_cost_eur) != set(grid_baseline_cost_eur):
        missing = set(p2p_cost_eur) ^ set(grid_baseline_cost_eur)
        raise ValueError(
            f"p2p and grid-baseline costs cover different households: {sorted(missing)}"
        )
    return [
        compute_household_outcome(
            household_id, p2p_cost_eur[household_id], grid_baseline_cost_eur[household_id]
        )
        for household_id in sorted(p2p_cost_eur)
    ]


def avg_savings_pct(outcomes: Sequence[HouseholdOutcome]) -> float:
    """Run-wide savings as a share of what the grid-only baseline would have cost.

    Weighted by each household's baseline, not a plain mean of the per-household
    percentages. A net seller's baseline is small — it only ever earned the export
    tariff — so its percentage is large, and a plain mean lets one small producer
    drag the whole run's "average saving" above 100%. The weighted figure is the one
    that survives being questioned.
    """
    reference_eur = sum(abs(outcome.total_cost_eur_grid_baseline) for outcome in outcomes)
    if reference_eur <= 0:
        return 0.0
    return sum(outcome.savings_eur for outcome in outcomes) / reference_eur * 100.0


def peak_load_reduction_pct(
    p2p_grid_import_kwh_per_tick: Sequence[float],
    grid_only_import_kwh_per_tick: Sequence[float],
) -> float:
    """Percentage cut in the community's worst simultaneous grid draw.

    Both series are community-wide grid imports in kWh, one entry per tick: what was
    actually drawn from the grid under P2P, and what would have been drawn with no
    marketplace at all. The comparison is peak against peak — the highest single
    tick of each — because peak load is what strains a distribution network, not the
    total. Returns 0.0 when the grid-only peak is zero (nothing to reduce), and is
    negative if P2P somehow drew more.
    """
    peak_grid_only_kwh = max(grid_only_import_kwh_per_tick, default=0.0)
    if peak_grid_only_kwh <= 0:
        return 0.0
    peak_p2p_kwh = max(p2p_grid_import_kwh_per_tick, default=0.0)
    return (peak_grid_only_kwh - peak_p2p_kwh) / peak_grid_only_kwh * 100.0


def co2_avoided_kg(
    traded_kwh: float, grid_co2_g_per_kwh: float = IE_GRID_CO2_G_PER_KWH
) -> float:
    """CO2 not emitted, in kg, because ``traded_kwh`` came from a neighbour not the grid.

    Assumes every traded kWh displaces a kWh of grid import at the grid's average
    carbon intensity, and that the neighbour's solar is zero-carbon at the margin.
    Both are simplifications worth stating out loud whenever this number is quoted.
    """
    if traded_kwh < 0:
        raise ValueError("traded_kwh must be >= 0")
    return traded_kwh * grid_co2_g_per_kwh / 1000.0


def compute_run_summary(
    run_id: str,
    strategy_name: str,
    outcomes: Sequence[HouseholdOutcome],
    num_ticks: int,
    traded_kwh: float,
    p2p_grid_import_kwh_per_tick: Sequence[float],
    grid_only_import_kwh_per_tick: Sequence[float],
    grid_co2_g_per_kwh: float = IE_GRID_CO2_G_PER_KWH,
) -> RunSummary:
    """Roll a whole run up into the one contract the demo actually quotes.

    ``traded_kwh`` is the run's total cleared volume; the two import series are
    community-wide grid draw per tick, P2P versus the no-marketplace counterfactual.
    ``num_households`` is taken from ``outcomes`` — every household that took part
    has exactly one.
    """
    return RunSummary(
        run_id=run_id,
        strategy_name=strategy_name,
        num_households=len(outcomes),
        num_ticks=num_ticks,
        total_savings_eur=sum(outcome.savings_eur for outcome in outcomes),
        avg_savings_pct=avg_savings_pct(outcomes),
        peak_load_reduction_pct=peak_load_reduction_pct(
            p2p_grid_import_kwh_per_tick, grid_only_import_kwh_per_tick
        ),
        co2_avoided_kg=co2_avoided_kg(traded_kwh, grid_co2_g_per_kwh),
    )
