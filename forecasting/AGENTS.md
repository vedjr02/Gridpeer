# AGENTS.md — forecasting/

Read the root `AGENTS.md` first — this file only covers what's specific to this module.

## Scope

Per-household, per-tick forecasts of demand and solar generation, feeding the agents' bidding decisions. This module does not decide strategy — it only predicts what's likely to happen, and reports how confident it is.

## Inputs / outputs (contracts, not internals)

- **Consumes:** historical household consumption + solar data (from `data/`)
- **Produces:** `ForecastOutput` (one per household per tick, consumed by `agents/` and displayed on `dashboard/`)

## Suggested build order

1. Start boring: a naive baseline (same time yesterday, or a rolling average) — this is your own sanity check before anything fancier, and it should stay in the codebase as a fallback.
2. LightGBM or Prophet for demand forecasting per household. Don't reach for a deep learning architecture first — with the data volumes here (CER trial households, half-hourly), a gradient-boosted model will likely outperform a neural net anyway, and it trains in seconds instead of needing GPU time the RL team also wants.
3. Solar generation forecasting: if using real weather-driven data (PVGIS), a simpler model may suffice since generation is largely deterministic given weather + panel capacity. Don't over-invest here relative to demand forecasting.
4. Confidence field: doesn't need to be sophisticated — a normalised recent-error metric is enough. It exists so the agents module *can* use it later (e.g. bid more conservatively under low confidence) — not a hard requirement for week 1.

## Backtesting

Hold out the last N days of the historical data per household and report standard forecasting error (MAE or MAPE) before wiring the model into the live pipeline. This becomes a genuinely reportable number in the final writeup, not just an internal sanity check.
