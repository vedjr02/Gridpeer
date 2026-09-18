"""Persist a baseline run and an RL run side by side, for the dashboard to compare.

    python -m agents.train_rl              # once: trains models/ppo_v1.zip
    python scripts/compare_strategies.py   # then: persists both runs
    streamlit run dashboard/app.py         # Strategy comparison tab: pick one of each

This lives outside the four modules on purpose. Persisting a run is the dashboard's
job and choosing a strategy is the agents module's; neither should import the
other, so the composition happens here, the way tests/ composes them. Both runs use
the dashboard's own demo households and ``run_pipeline``, so the comparison is on
identical households, ticks and settlement.
"""

from __future__ import annotations

from agents.baseline import STRATEGY_NAME as BASELINE_NAME
from agents.rl_policy import DEFAULT_MODEL_PATH, DEFAULT_STRATEGY_NAME, PPOTrader
from dashboard.orchestrator import DEFAULT_DB_PATH, demo_households, run_pipeline


def main() -> None:
    """Run both strategies on the dashboard's households and persist each run."""
    households = demo_households()
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    runs = [
        (BASELINE_NAME, None),
        (DEFAULT_STRATEGY_NAME, PPOTrader(model_path=DEFAULT_MODEL_PATH)),
    ]
    summaries = []
    for name, strategy in runs:
        result = run_pipeline(
            households, strategy=strategy, strategy_name=name, db_path=DEFAULT_DB_PATH
        )
        summaries.append(result.summary)
        print(
            f"{name:<22} EUR {result.summary.total_savings_eur:6.3f} saved  "
            f"({result.summary.avg_savings_pct:5.1f}%)  -> {result.summary.run_id}"
        )

    baseline, policy = summaries
    uplift = (
        (policy.total_savings_eur - baseline.total_savings_eur)
        / abs(baseline.total_savings_eur)
        * 100.0
        if baseline.total_savings_eur
        else 0.0
    )
    print(f"uplift on the dashboard's demo households: {uplift:+.1f}%")
    print(f"both persisted to {DEFAULT_DB_PATH}; open the Strategy comparison tab to view")


if __name__ == "__main__":
    main()
