# agents/

Owner: Amogh Gaikwad (@ag2502)

Baseline + RL trading strategies. See `CLAUDE.md` in this folder for build guidance.

## Status

- [x] Rule-based baseline, wired through the real pipeline
- [x] PettingZoo multi-agent environment (`rl_env.py`) + SB3 shared-policy adapter (`rl_vec_env.py`)
- [x] PPO training (Stable-Baselines3) — `ppo_v1`: **+11.9% vs baseline on unseen data, every household better off**
- [x] Evaluation harness producing `RunSummary` (baseline vs. RL)

## Running just this module

```bash
python -m agents.run_baseline         # baseline: forecasting -> agents -> market -> settlement
python -m agents.train_rl             # train ppo_v1 (~1.5 min), select on validation, report on test
python scripts/compare_strategies.py  # persist baseline + ppo_v1 runs for the dashboard
python -m agents.battery_experiment   # Stage 5: battery-controlling policy (~5 min)
pytest agents/
```

## Result: ppo_v1 vs the rule-based baseline

Scored once on a test set never used for training or selection: 20 synthetic two-day
runs of 4 households, from a cold start, as the dashboard runs.

| | Community savings | Runs won | Worst household vs baseline |
|---|---|---|---|
| Rule-based baseline | EUR 11.39 | — | — |
| **ppo_v1** | **EUR 12.75 (+11.9%)** | **20 / 20** | **+EUR 0.09 (all better off)** |

Five training seeds, chosen on a separate validation set with fairness first (no
household worse off), then uplift. On the dashboard's own single demo run the gain is
smaller (+3.1%) and two households move by about a cent each way — one run is noisy,
and its producer has a 3 kW panel where training used 6 kW. Fairness is established
across runs, not guaranteed in every run.

**What it learned:** when to offer more or less than the naive forecast says. The naive
forecast lags the sun, so its errors are predictable from the time of day; that is the
headroom (+56% against a perfect-foresight ceiling). Price stays with the baseline rule
— see `rl_env.py` for the measurements behind every design choice, including the two
reward designs that failed.

## Stage 5: agents that control the battery

`python -m agents.battery_experiment` — measured against the **status quo** (no battery,
no market), because settlement's grid-only figure is taken after the battery and cannot
see what a battery earns. Test set: 20 unseen two-day runs, 4 households, 2.5 kW batteries.

| Strategy | Savings vs status quo | Evening peak |
|---|---|---|
| Baseline, no battery (the dashboard today) | EUR 11.39 | 0% |
| Baseline + automatic battery, aware of it | EUR 55.47 | -19.4% |
| **ppo_battery_v1** (controls its battery) | **EUR 61.68 (+11.2%)** | -19.4% |

Every household beats the status quo. The household with no solar gains the most
(EUR 4.96 -> 11.93): the automatic battery soaks up midday solar the consumer used to
buy, and the policy sells stored energy back to it. **The trade-off:** the two prosumers
earn EUR 3.65-4.71 less than under the automatic battery, because a producer's stored
energy now competes for the same evening demand. Whether "everyone beats the status
quo" or "nobody earns less than under the automatic battery" is the right rule is a
team decision; the experiment prints both. The extra peak cut seen on validation did
not hold on test — the -19.4% comes from the batteries themselves.

Two lessons that shaped it: an absolute battery target lost 78% (it had to rediscover
the automatic battery), so the action is a correction to it; and without
potential-based reward shaping PPO unlearned charging (-70%), because storing at noon
costs money now and pays hours later.

**Deployed (schema 0.2.0).** Agents now receive the battery state (`HouseholdState`) and
adjust the battery through `AgentDecision.battery_offset_kwh`. Through
`dashboard.run_pipeline(use_battery=True, battery_power_kw=2.5)` the policy reproduces
the experiment exactly (EUR 61.70 vs 55.47 on the 20 test runs; pinned by
`tests/test_battery_pipeline.py` for the baseline). `BatteryPPOTrader` serves it, and
`python scripts/compare_strategies.py` persists the battery pair for the dashboard. On
the dashboard's demo households: EUR 2.53 vs 2.22 (+13.8%), peak -12.8%, every household
saving. Given the battery state, `RuleBasedTrader` plans around the automatic battery; with
batteries off it behaves exactly as before.

The offset is an *adjustment* to the automatic battery, not the target setpoint the
proposal first described: sent as a setpoint, the same policy fell from +11.2% to -8.5%
against the automatic battery, because a setpoint stops the battery absorbing forecast
error.

## The RL environment

`GridPeerParallelEnv` follows PettingZoo's Parallel API, one agent per household, on top of
`simulation.MarketSimulator` — the same clearing and meter settlement the dashboard runs.

- **Observes** its own forecast, the time of day and its own tariffs: only what
  `decide(forecast, profile)` receives, so a trained policy deploys into the pipeline as is.
- **Chooses** how much of its forecast position to offer and how eagerly to price it inside
  its tariff gap. The baseline is a point in this action space, so a policy can always
  recover it.
- **Is rewarded** with meter-settled savings against grid-only: over-offering energy it
  does not have is bought back at the import tariff, not paid for.

Driven with the baseline's decisions it reproduces `run_pipeline`'s savings exactly — that
equivalence is a test, so the policy is trained on the market it is judged on.
`SharedPolicyVecEnv` presents every household as one slot of an SB3 vector env: one shared
policy (parameter sharing), which is also how it is deployed.

Batteries are off by default because no shared contract lets an agent observe them yet —
see the open `schema-change` proposal.

## How the baseline prices an order

Side and quantity come from the forecast net position. The price is interpolated
inside that household's own tariff gap — between what the grid pays for an export and
what it charges for an import, the only margin a P2P trade can capture. "Eagerness"
slides the price within the gap and rises with the size of the position, because a
large surplus is worth conceding margin for rather than dumping to the grid. Both
sides using this rule always produce a crossing book; a baseline that rarely traded
would flatter every RL result measured against it.

## Settled at the meter, not at the order

Every tick's cost and grid load come from what each household actually generated and
consumed; the difference between that and what it traded is bought or sold at its own
grid tariffs. A strategy that offers energy it does not have pays for the shortfall
instead of booking it as a saving — the first thing an RL policy would otherwise learn.

**Known artifact:** the forecaster has no history at tick 0, so nobody offers a sale in
it. Because peak-load reduction is a worst-tick measure, a run whose demand is flat
reports 0% reduction on the strength of that one untradeable tick. Warming the
forecaster on history before the measured run is the fix; it is not done yet.

## Two open questions for the team

**Batteries take no part in a run yet.** Households trade their forecast net position
— solar minus demand, no storage in between. `simulation/` has a working battery
model, but wiring it in means first deciding whether the battery is environment-owned
physics or something the agent controls: the forecaster predicts raw demand and solar,
so right now nothing owns that choice. Today's savings figures are the no-storage
floor, and this is the likeliest single source of improvement in them.

**`RunSummary` is claimed by two modules.** `agents/CLAUDE.md` says this harness
produces it; `dashboard/CLAUDE.md` says the dashboard does. It is computed here
because importing the dashboard would break the module boundary, but the arithmetic
should live in one place before two versions drift apart.

## A caveat on per-household savings %

`HouseholdOutcome.savings_pct` is a percentage of a household's grid-only baseline, so
a net seller — whose baseline is only ever the low export tariff — can show a figure
well over 100%. `RunSummary.avg_savings_pct` is therefore weighted by baseline cost
rather than being a plain mean of those percentages, which one small producer would
otherwise dominate.
