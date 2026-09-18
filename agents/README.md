# agents/

Owner: Amogh Gaikwad (@ag2502)

Baseline + RL trading strategies. See `CLAUDE.md` in this folder for build guidance.

## Status

- [x] Rule-based baseline, wired through the real pipeline
- [ ] PettingZoo-style multi-agent environment wrapper
- [ ] PPO training (Stable-Baselines3)
- [x] Evaluation harness producing `RunSummary` (baseline vs. RL)

## Running just this module

```bash
python -m agents.run_baseline    # forecasting -> agents -> market -> settlement, printed
pytest agents/
```

## How the baseline prices an order

Side and quantity come from the forecast net position. The price is interpolated
inside that household's own tariff gap — between what the grid pays for an export and
what it charges for an import, the only margin a P2P trade can capture. "Eagerness"
slides the price within the gap and rises with the size of the position, because a
large surplus is worth conceding margin for rather than dumping to the grid. Both
sides using this rule always produce a crossing book; a baseline that rarely traded
would flatter every RL result measured against it.

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
