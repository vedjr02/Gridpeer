"""Persist baseline and RL runs side by side, for the dashboard to compare.

    python -m agents.train_rl              # once: trains models/ppo_v1.zip
    python -m agents.battery_experiment    # once: trains models/ppo_battery_v1.zip
    python scripts/compare_strategies.py   # then: persists every run
    streamlit run dashboard/app.py         # Strategy comparison tab: pick a pair

Two pairs, each on identical households and ticks:

- batteries off: ``rule_based_baseline`` vs ``ppo_v1``;
- batteries on (2.5 kW): ``rule_based_baseline+battery`` vs ``ppo_battery_v1``. The
  baseline here is handed the battery state (schema 0.2.0) and plans around it;
  ``ppo_battery_v1`` also adjusts its battery through ``battery_offset_kwh``.

This lives outside the four modules on purpose. Persisting a run is the dashboard's
job and choosing a strategy is the agents module's; neither should import the
other, so the composition happens here, the way tests/ composes them. A pair whose
trained model is missing is skipped with a note.
"""

from __future__ import annotations

from agents.baseline import STRATEGY_NAME as BASELINE_NAME
from agents.rl_policy import (
    BATTERY_MODEL_PATH,
    BATTERY_STRATEGY_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_STRATEGY_NAME,
    BatteryPPOTrader,
    PPOTrader,
)
from dashboard.orchestrator import DEFAULT_DB_PATH, demo_households, run_pipeline

BATTERY_POWER_KW = 2.5


def _persist(households, strategy, name: str, battery: bool):
    """Run one strategy through the dashboard pipeline and persist it; return its summary."""
    result = run_pipeline(
        households,
        strategy=strategy,
        strategy_name=name,
        db_path=DEFAULT_DB_PATH,
        use_battery=battery,
        battery_power_kw=BATTERY_POWER_KW if battery else None,
    )
    summary = result.summary
    print(
        f"  {name:<30} EUR {summary.total_savings_eur:6.3f} saved "
        f"({summary.avg_savings_pct:5.1f}%), peak {summary.peak_load_reduction_pct:+5.1f}% "
        f"-> {summary.run_id}"
    )
    return summary


def _uplift(baseline, candidate) -> float:
    base = baseline.total_savings_eur
    return (candidate.total_savings_eur - base) / abs(base) * 100.0 if base else 0.0


def main() -> None:
    """Persist both pairs on the dashboard's demo households."""
    households = demo_households()
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("batteries off", DEFAULT_MODEL_PATH, False, BASELINE_NAME,
         lambda: PPOTrader(model_path=DEFAULT_MODEL_PATH), DEFAULT_STRATEGY_NAME),
        ("batteries on", BATTERY_MODEL_PATH, True, f"{BASELINE_NAME}+battery",
         lambda: BatteryPPOTrader(model_path=BATTERY_MODEL_PATH), BATTERY_STRATEGY_NAME),
    ]
    for label, model_path, battery, base_name, make_policy, policy_name in pairs:
        print(f"{label}:")
        if not model_path.exists():
            print(f"  skipped: no trained model at {model_path}")
            continue
        baseline = _persist(households, None, base_name, battery)
        policy = _persist(households, make_policy(), policy_name, battery)
        print(f"  uplift on the dashboard's demo households: {_uplift(baseline, policy):+.1f}%")
    print(f"persisted to {DEFAULT_DB_PATH}; open the Strategy comparison tab to view")


if __name__ == "__main__":
    main()
