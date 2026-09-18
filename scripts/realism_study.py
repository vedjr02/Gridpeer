"""Stage 6: do the results survive scale, winter and network charges?

    python -m agents.train_rl                # once, for ppo_v1
    python -m agents.battery_experiment      # once, for ppo_battery_v1
    python scripts/realism_study.py          # writes reports/realism_study.md

Three stress tests of every strategy, all on the test split (never trained on),
scored against the status quo (no battery, no market):

1. **Scale** — 4 to 100 households. Does the auction still clear sensibly, do the
   policies (trained on 4 households) still help, and how fast is a tick?
2. **Season** — solar scaled to 50% and 20% of the synthetic generator's output.
   ``data.synthetic`` has no seasons: every day gets the same sun apart from
   clouds. Scaling is an *assumption* standing in for an Irish winter, where PV
   yield is a fraction of summer's — not data. Real CER + PVGIS series replace it.
3. **Network charges** — a per-kWh charge on every P2P trade, paid by the buyer.
   The grid import tariff already carries network charges; leaving them off P2P
   trades overstates the saving. Strategies are *not* re-optimised for the charge,
   so the figures show what today's policies would face.

Real CER smart-meter data is not here: it needs an ISSDA access request.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO

from agents.battery_experiment import (
    BATTERY_MODEL_PATH,
    Decider,
    baseline_decider,
    policy_decider,
    rollout,
    without_batteries,
)
from agents.rl_policy import DEFAULT_MODEL_PATH
from agents.train_rl import EPISODE_DAYS, TEST_SEED_BASE, cohort
from simulation.simulator import HouseholdSeries

REPORT_PATH = Path("reports/realism_study.md")


@dataclass(frozen=True)
class Strategy:
    name: str
    decide: Decider
    battery: bool


def strategies() -> list[Strategy]:
    """Every strategy with a trained model available, plus the two baselines."""
    found = [
        Strategy("baseline, no battery", baseline_decider(aware=False), False),
        Strategy("baseline + auto battery (aware)", baseline_decider(aware=True), True),
    ]
    if DEFAULT_MODEL_PATH.exists():
        model = PPO.load(DEFAULT_MODEL_PATH, device="cpu")
        found.insert(1, Strategy("ppo_v1, no battery", policy_decider(model, False, "ppo_v1"),
                                 False))
    if BATTERY_MODEL_PATH.exists():
        model = PPO.load(BATTERY_MODEL_PATH, device="cpu")
        found.append(Strategy("ppo_battery_v1", policy_decider(model, True, "ppo_battery_v1"),
                              True))
    return found


def scaled_solar(households: Sequence[HouseholdSeries], factor: float) -> list[HouseholdSeries]:
    """The same households with solar output multiplied by ``factor``."""
    return [
        HouseholdSeries(profile=h.profile, demand_kwh=h.demand_kwh,
                        solar_kwh=[kwh * factor for kwh in h.solar_kwh])
        for h in households
    ]


@dataclass
class Row:
    savings_eur_per_household_day: float
    peak_reduction_pct: float
    trades_per_tick: float
    ms_per_tick: float


def measure(
    strategy: Strategy,
    num_households: int,
    cohorts: int,
    transform: Callable[[list[HouseholdSeries]], list[HouseholdSeries]] = lambda h: h,
    network_charge_eur_per_kwh: float = 0.0,
) -> Row:
    """Average one strategy over ``cohorts`` unseen runs."""
    savings = peak = trades = seconds = 0.0
    ticks = 0
    for k in range(cohorts):
        households = transform(cohort(num_households, EPISODE_DAYS, TEST_SEED_BASE + k))
        if not strategy.battery:
            households = without_batteries(households)
        counted = _CountingDecider(strategy.decide)
        started = time.perf_counter()
        outcome = rollout(households, counted, network_charge_eur_per_kwh)
        seconds += time.perf_counter() - started
        savings += outcome.savings_eur
        peak += outcome.peak_reduction_pct
        trades += counted.orders
        ticks += counted.ticks
    household_days = num_households * EPISODE_DAYS * cohorts
    return Row(
        savings_eur_per_household_day=savings / household_days,
        peak_reduction_pct=peak / cohorts,
        trades_per_tick=trades / ticks,
        ms_per_tick=seconds / ticks * 1000.0,
    )


class _CountingDecider:
    """Wraps a decider to count orders and ticks without changing what it does."""

    def __init__(self, decide: Decider) -> None:
        self.decide, self.orders, self.ticks = decide, 0, 0

    def __call__(self, simulator, forecasts):
        orders, battery = self.decide(simulator, forecasts)
        self.orders += len(orders)
        self.ticks += 1
        return orders, battery


def main() -> None:
    """Run all three studies and write the report."""
    found = strategies()
    lines = [
        "# Stage 6 — realism study",
        "",
        "Generated by `python scripts/realism_study.py`. Test split only (never trained on); "
        "savings are against the status quo (no battery, no market), in EUR per household "
        "per day. Synthetic data throughout — see the caveats at the end.",
        "",
    ]

    lines += ["## 1. Scale", "",
              "Policies were trained on 4 households and applied unchanged. "
              "Mean of 5 unseen two-day runs per size.", "",
              "| Households | Strategy | EUR / household / day | Peak cut | Orders / tick "
              "| ms / tick |", "|---|---|---|---|---|---|"]
    for n in (4, 8, 20, 50, 100):
        for s in found:
            r = measure(s, n, cohorts=5)
            lines.append(f"| {n} | {s.name} | {r.savings_eur_per_household_day:.3f} | "
                         f"{r.peak_reduction_pct:.1f}% | {r.trades_per_tick:.1f} | "
                         f"{r.ms_per_tick:.1f} |")
        print(f"scale {n} done")

    lines += ["", "## 2. Season (solar scaled — an assumption, not data)", "",
              "4 households, mean of 20 unseen two-day runs.", "",
              "| Solar | " + " | ".join(s.name for s in found) + " |",
              "|---|" + "---|" * len(found)]
    for factor in (1.0, 0.5, 0.2):
        rows = [measure(s, 4, 20, lambda h, f=factor: scaled_solar(h, f)) for s in found]
        cells = [f"{r.savings_eur_per_household_day:.3f}" for r in rows]
        lines.append(f"| {factor:.0%} | " + " | ".join(cells) + " |")
        print(f"season {factor} done")

    lines += ["", "## 3. Network charge on P2P trades (buyer pays)", "",
              "4 households, mean of 20 unseen two-day runs. The tariff gap is "
              "EUR 0.18/kWh (0.25 import - 0.07 export).", "",
              "| Charge EUR/kWh | " + " | ".join(s.name for s in found) + " |",
              "|---|" + "---|" * len(found)]
    for charge in (0.0, 0.02, 0.05, 0.09, 0.12, 0.18):
        rows = [measure(s, 4, 20, network_charge_eur_per_kwh=charge) for s in found]
        cells = [f"{r.savings_eur_per_household_day:.3f}" for r in rows]
        lines.append(f"| {charge:.2f} | " + " | ".join(cells) + " |")
        print(f"charge {charge} done")

    lines += [
        "", "## Caveats", "",
        "- **Synthetic data.** Every number comes from `data.synthetic`, not Irish "
        "smart-meter data. Real CER data needs an ISSDA access request.",
        "- **No seasons in the generator.** The season table scales solar uniformly; a "
        "real winter also changes demand (heating, longer evenings).",
        "- **Policies not re-optimised** for scale, season or network charges — the tables "
        "show how today's trained policies generalise, not the best achievable.",
        "- **Flat tariffs.** One import and one export price for everyone, all day. "
        "Time-of-use tariffs would change the battery results most.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
