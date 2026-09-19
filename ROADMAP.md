# Roadmap

## Where we are — 18 September 2026

Week 1 and most of week 2 are done: the pipeline is real end to end, not a skeleton.
`make run-sim` forecasts every household from history only, asks a strategy for an
order, clears a per-tick double auction, settles against metered positions,
persists the run to SQLite and rolls it up into `RunSummary`. `streamlit run
dashboard/app.py` renders that persisted run — savings, CO2 avoided, peak-load
reduction, forecast diagnostics.

The two things standing between here and week 3:

1. **RL agents have not started.** Everything they need is in place: `run_pipeline`
   takes any object with `decide(forecast, profile) -> AgentDecision | None`, so a
   trained policy plugs in without touching the loop, and the dashboard's strategy
   comparison tab fills in as soon as a second run is persisted under a different
   `strategy_name`.
2. **The data is still synthetic.** `data/synthetic.py` produces series in the shape
   the CER loader will use, so the swap is one file — but every savings number quoted
   today comes from generated data, not Irish smart-meter data.

Open on GitHub: PR #19 fixes a semantic clash between the pipeline and forecasting
merges that currently leaves `main` failing 21 tests; PR #20 renames the AI context
files. Merge #19 before anyone builds on `main`.

Four weeks, four people, all skilled, all with time. This is deliberately front-loaded: the goal is a real end-to-end pipeline by the end of week 1, so weeks 2–4 are about improving real modules instead of assembling them for the first time under deadline pressure.

## Deciding roles (do this before Day 1)

Pick one owner per module. Suggested fit, not a rule:
- **Simulation & market mechanism** — whoever's strongest on algorithmic/systems thinking; this module has the least ML and the most careful logic.
- **RL agents** — whoever's most excited about RL specifically. This is the hardest module to make actually work well, and enthusiasm matters more here than anywhere else.
- **Forecasting** — whoever's strongest on classical ML / time series.
- **Orchestration & dashboard** — good seat for the person who cares most about the outcome narrative (savings %, CO2 avoided) rather than pure model metrics — often the person driving the BA/product framing.

Also settle now: **who has final say when two people disagree on a schema field.** Small decision, but it's the argument that stalls week 1 in projects like this if it's not settled up front.

## Week 1 — Contracts and an ugly end-to-end pipeline

**Day 1–2**
- [x] Agree the shared interfaces as actual typed schemas (already scaffolded in `shared/schemas.py` — review and adjust together, don't skip this review just because it's pre-written)
- [ ] Fill in `README.md` team table with real names
- [x] Repo pushed to GitHub, all four have push access
- [ ] `make setup && make test` passes on all four laptops
- [ ] One issue per person on the GitHub Project board, scoped to "build the skeleton of my module"

**Day 3–5**
- [x] Each person builds a skeleton of their module that satisfies the shared contract using fake/placeholder data — simulation emits dummy trades, dashboard renders them, forecaster returns placeholder numbers, agent returns a hardcoded decision
- [x] **Goal by end of week 1: a fully wired, end-to-end pipeline that is ugly but real.** This is the single highest-leverage milestone in the whole project — it converts "four modules that might integrate eventually" into "one working system we're now improving."

## Week 2 — Real modules

- [~] **Simulation:** per-tick double auction, household environment and settlement are implemented and tested; calibration against real CER data is still outstanding — households come from `data/synthetic.py`
- [x] **Forecasting:** LightGBM demand model with a walk-forward backtest (MAE/MAPE); solar uses a tick-of-day climatology — boring and reliable beats fancy and unstable at this stage
- [x] **Agents:** the rule-based baseline runs through the real pipeline and is the comparison baseline for everything after
- [x] **Dashboard:** reads persisted runs only, with the outcomes narrative split across Outcomes / Strategy comparison / Forecast detail

## Week 3 — Integration and the actual result

- [x] Full integration of all real modules end to end
- [ ] RL agent training: reward shaping, training stability
- [ ] **The core deliverable of the whole project:** RL-agent performance vs. rule-based baseline, measured and visualised. This comparison is the entire demo story. The harness and the comparison view are built and waiting on a persisted RL run.
- [x] Add business-impact metrics to the dashboard: household cost savings %, CO2 avoided, peak-load reduction vs. grid-only baseline
- [x] Start writing tests and the README as you go — 185 tests, one file per module plus the cross-module contract tests

## Week 4+ — Polish, and where "keep building for a month" pays off

Pick based on appetite, don't try to do all of them:
- [ ] Scenario simulator: heatwave, EV charging surge, grid outage — how does the market behave under stress?
- [ ] Scale to more households, check the market mechanism still clears sensibly at scale
- [ ] Dynamic pricing under grid-stress events
- [ ] Real deployment — Streamlit Cloud or a small VPS, so it's a live link, not just a repo
- [ ] Swap dashboard to FastAPI + React now that the underlying system is stable
- [ ] Write a short case-study document framed as a business outcome ("agents that learn to trade save households X% more than the naive baseline"), not just model metrics — this is what makes it read differently from a standard ML-only submission in an interview

## Definition of done for the project as a whole

You can point at a single number and defend it: "our RL agents produced X% more household savings than the rule-based baseline, on real Irish smart-meter data, over N simulated days." Everything else in the roadmap exists to make that sentence true and demonstrable.
