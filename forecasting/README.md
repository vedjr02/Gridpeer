# forecasting/

Owner: _TBD_

Per-household demand + solar generation forecasting. See `CLAUDE.md` in this folder for build guidance.

## Status

- [x] Naive baseline forecaster (`baseline.NaiveForecaster`, rolling average + flat fallback)
- [ ] LightGBM/Prophet demand model
- [ ] Solar generation forecaster
- [x] Backtest report (MAE/MAPE per household) — `backtest.backtest_household`, walk-forward

## Running just this module

```bash
python -m forecasting.backtest    # walk-forward MAE/MAPE demo report
pytest forecasting/
```
