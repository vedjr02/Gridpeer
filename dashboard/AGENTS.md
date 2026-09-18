# AGENTS.md — dashboard/

Read the root `AGENTS.md` first — this file only covers what's specific to this module.

## Scope

This module has two jobs that are easy to conflate but should stay separated in the code:

1. **Orchestration** — the loop that advances the simulation tick by tick: call `forecasting/`, pass results to `agents/`, pass agent decisions to `simulation/`, persist the resulting `TradeEvent`/`MarketState`. This is plumbing, keep it boring and well-tested.
2. **Dashboard/visualisation** — turning persisted history into the outcomes narrative: price over time, trade volume, per-household savings vs. grid-only baseline, CO2 avoided, peak-load reduction. This is the part that actually gets looked at in a demo, so it deserves real design thought, not a default Streamlit table dump.

## Inputs / outputs (contracts, not internals)

- **Consumes:** everything (`ForecastOutput`, `AgentDecision`, `TradeEvent`, `MarketState` from the other three modules)
- **Produces:** `HouseholdOutcome` and `RunSummary` (the rollups other modules don't need but the dashboard and final writeup do)

## Suggested build order

1. Orchestration loop first, wired to whatever skeleton modules exist in week 1 — this is what makes "ugly end-to-end pipeline by end of week 1" possible.
2. Persistence: SQLite or DuckDB, one table roughly per shared model. Don't over-engineer the schema — this is for your own team's iteration speed, not a production system.
3. Streamlit dashboard, weeks 1–3: a few clear charts beat many cluttered ones. Priority order: (a) cumulative household savings, P2P vs. grid baseline, (b) price over time, (c) trade volume, (d) CO2 avoided / peak-load reduction as headline numbers, not buried in a table.
4. Week 4+, optional: FastAPI + React rebuild, once the underlying data isn't changing shape weekly.

## The narrative this dashboard needs to tell

Not "here are some charts of a simulation." The story is: *households using this marketplace saved X% versus just exporting to the grid, and the RL-trained agents did better than the naive rule-based ones by Y%.* Every chart should serve one of those two numbers, directly or as supporting evidence. If a chart doesn't clearly support either claim, it's probably not earning its place on the main view — move it to a "details" tab instead of the headline.
