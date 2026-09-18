# simulation/

Owner: Amogh Gaikwad (@ag2502)

Market clearing mechanism (continuous double auction) + household environment. See `AGENTS.md` in this folder for build guidance, and the root `README.md` for how this fits the overall architecture.

## Status

- [x] 2-household deterministic test case
- [~] Household environment — battery + net position done; real data loading waits on `data/`
- [x] N-household continuous double auction
- [x] Grid-baseline fallback for unmatched orders

## Running just this module

```bash
pytest simulation/
```

## The three pieces

- `market.py` — `clear_tick()`: the auction. Orders in, trades out.
- `environment.py` — `HouseholdEnvironment`: steps a household through its demand/solar
  series, running the battery, and reports the net position actually left to trade. Series
  are passed in; the loader that reads real CER/PVGIS data belongs in `data/`.
- `settlement.py` — `settle_tick()`: what the tick cost each household under P2P versus the
  grid-only counterfactual. Unmatched orders fall back to that household's own grid tariffs
  and therefore save nothing, which is what keeps the headline savings figure honest.
  **Pass `actual_net_position_kwh`** (each household's metered net position) whenever you
  have it: the gap between what a household traded and what its meter recorded is then
  settled at grid tariffs, so forecast error costs money and selling energy you never
  generated cannot turn a profit. Without it, settlement trusts the orders — on the demo
  households that overstates the rule-based baseline's savings by about a quarter.

Costs are signed: positive means money left the household, so a seller's cost is negative.
`HouseholdOutcome` and `RunSummary` are the dashboard's to build by summing these across a
run — no schema change needed.

## Clearing, in one paragraph

`clear_tick(tick, timestamp, orders)` takes a tick's `AgentDecision` book and returns a
`MarketState`. Buys rank by descending limit price, sells by ascending, ties broken by
arrival order in the input sequence (price-time priority). The best buy matches the best
sell while the buyer's maximum is at least the seller's minimum; each match trades the
smaller quantity at the midpoint of the two limit prices, and the larger order keeps its
remainder on the book. Unmatched orders — including a partial fill's remainder — come back
in `MarketState` for the caller to settle against grid tariffs. The tick's
`clearing_price_eur_per_kwh` is the volume-weighted average of its trades, or `None` when
nothing cleared. Clearing is deterministic: the same book always produces the same result.
