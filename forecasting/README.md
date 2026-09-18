# forecasting/

Owner: _TBD_

Per-household demand + solar generation forecasting. See `CLAUDE.md` in this folder for build guidance.

## Status

- [x] Naive baseline forecaster (`baseline.NaiveForecaster`, rolling average + flat fallback)
- [x] LightGBM demand model (`gbm.GBMForecaster`, refits as history accrues)
- [ ] Solar generation forecaster
- [x] Backtest report (MAE/MAPE per household) — `backtest.backtest_models`, rolling-origin holdout comparing naive vs GBM

## Running just this module

```bash
python -m forecasting.backtest    # holdout MAE/MAPE table, naive vs GBM
pytest forecasting/
```
