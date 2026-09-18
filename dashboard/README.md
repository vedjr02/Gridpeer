# dashboard/

Owner: _TBD_

Orchestration loop + live outcomes dashboard. See `AGENTS.md` in this folder for build guidance.

## Status

- [x] Orchestration loop (`orchestrator.py`) — full pipeline: forecasting → agents → simulation → persistence → rollups
- [x] Persistence layer (`persistence.py`, SQLite) — profiles, forecasts, decisions, trades, market states, outcomes, summaries
- [x] Streamlit dashboard (`app.py`) — Outcomes / Strategy comparison / Forecast detail, all read from persisted runs
- [ ] (Week 4+, optional) FastAPI + React rebuild

## Running just this module

```bash
python -m dashboard.orchestrator  # run the pipeline, persist to data/gridpeer.db, print the summary
streamlit run dashboard/app.py    # dashboard over the persisted runs
pytest dashboard/
```

`make run-sim` runs the first command. Both sides read `GRIDPEER_DB` and default to
`data/gridpeer.db`, so a run written by one is the run the other renders. Run
databases are local artefacts and are gitignored.

## Plugging in a strategy

`run_pipeline` takes any object with a `decide` method — the `TradingStrategy`
protocol in `orchestrator.py`. The rule-based baseline is the default; a trained
policy drops in without touching the loop:

```python
from dashboard.orchestrator import demo_households, run_pipeline

result = run_pipeline(
    demo_households(),
    strategy=MyPolicy(),          # decide(forecast: ForecastOutput, profile: HouseholdProfile) -> AgentDecision | None
    strategy_name="ppo_v1",
    db_path="data/gridpeer.db",
)
print(result.summary.total_savings_eur, result.summary.avg_savings_pct)
```

Persist a baseline run and a policy run to the same database and the dashboard's
Strategy comparison tab puts them side by side.
