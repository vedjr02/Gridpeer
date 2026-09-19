# simulation/

Owner: Amogh Gaikwad (@ag2502)

Market clearing mechanism (a double auction cleared once per tick — a *call* auction, not a continuous one) + household environment. See `CLAUDE.md` in this folder for build guidance, and the root `README.md` for how this fits the overall architecture.

## Status

- [x] 2-household deterministic test case
- [~] Household environment — battery + net position done; real data loading waits on `data/`
- [x] N-household double auction, cleared once per tick
- [x] Grid-baseline fallback for unmatched orders
- [x] `MarketSimulator.step()` — physics, clearing and settlement in one call

## Running just this module

```bash
pytest simulation/
```

## Start here: `MarketSimulator`

Other modules should call this and nothing else. It is the only place that knows the
order a tick happens in.

```python
from simulation.simulator import HouseholdSeries, MarketSimulator

simulator = MarketSimulator([HouseholdSeries(profile, demand_kwh, solar_kwh), ...])
result = simulator.step(orders)          # orders: this tick's AgentDecision book

result.market_state      # the shared contract: trades, unmatched orders, price
result.settlements       # per household: P2P cost vs grid-only cost, and savings
result.household_states  # demand, solar, battery level, metered net position
result.savings_eur       # what the marketplace saved the community this tick
```

`reset()` replays a run from tick 0 with the starting charge. The same inputs always
produce the same outputs, across replays and across instances.

**Order within a tick, and why:** physics runs first (what a household generated,
used and stored is true whatever it bid), then the book clears, then the tick is
settled against the metered net position. Anything traded but not delivered — or
bought but not needed — is bought or sold at that household's grid tariffs.

**Speed:** about 0.33 ms per step with 100 households (~3,000 steps/s) and 1.7 ms
with 500, measured on Python 3.9; 3.11+ is faster. Roughly 1M training steps in
6 minutes at 100 households. If that ever becomes the bottleneck, the first lever is
building `TradeEvent` with `model_construct` to skip validation in the hot loop.

## Team decision: the battery is environment-owned physics

Agreed default, open to revisiting: surplus charges the battery before anything
reaches the market, and a deficit draws from it before anything is bought, so agents
trade only what the battery could not absorb or supply. It keeps the RL action space
to price and quantity, which is the difference between a learning problem that works
in three weeks and one that does not. Making charge/discharge an agent decision later
adds a field to `AgentDecision`; it does not change `step()`.

## The pieces

- `simulator.py` — `MarketSimulator.step()`: the entry point above, and the tick clock
  (`TICK_MINUTES`, `EPOCH`, `timestamp_for`) the whole pipeline should share.
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

## Battery control

`HouseholdEnvironment.step()` runs the automatic battery by default. Two optional
controls, both within capacity and an optional power limit (`max_battery_power_kw`, off
by default so every existing result is unchanged):

- `battery_offset_kwh` adjusts the automatic behaviour: positive discharges extra
  (stored energy to sell), negative holds charge back.
- `battery_setpoint_kwh` drives the charge toward a target level instead.

`planned_net_kwh()` runs the same physics on a forecast, so a household can plan an
order on the position its battery choice will actually leave — one function serves
both, and a test pins that planning on the true position predicts the meter exactly.
`MarketSimulator.step()` passes `battery_offsets_kwh` / `battery_setpoints_kwh` through.
`battery_flow_kwh()` is the physics as a pure function — the simulator steps with it and
agents plan with it — and `HouseholdEnvironment.contract_state()` emits the shared
`HouseholdState` (schema 0.2.0) that agents receive each tick.

## Network charges

`settle_tick(..., network_charge_eur_per_kwh=...)` levies a per-kWh charge on every P2P
trade (buyer pays; `network_charge_seller_share` moves part to the seller), and
`MarketSimulator` takes the same parameter. It defaults to zero, which is every other
result in the repo. A neighbour-to-neighbour trade still runs over the distribution
network, and the grid import tariff already carries those charges — so without this,
P2P savings are optimistic. `reports/realism_study.md` shows how fast they shrink.

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
