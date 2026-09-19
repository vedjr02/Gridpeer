# simulation/

Owner: Amogh Gaikwad (@ag2502)

Market clearing mechanism (a double auction cleared once per tick — a *call* auction, not a continuous one) + household environment. See `CLAUDE.md` in this folder for build guidance, and the root `README.md` for how this fits the overall architecture.

## Status

- [x] 2-household deterministic test case
- [~] Household environment — battery + net position done; real data loading waits on `data/`
- [x] N-household double auction, cleared once per tick
- [x] Grid-baseline fallback for unmatched orders
- [x] `MarketSimulator.step()` — physics, clearing and settlement in one call
- [x] Fair tie-breaking — tied orders share volume pro-rata, never by list position
- [x] Two pricing rules — pairwise midpoint (default) and uniform price
- [x] Property-based tests of the clearing and settlement invariants

## Running just this module

```bash
pytest simulation/                                                   # ~1s
HYPOTHESIS_PROFILE=thorough pytest simulation/test_market_properties.py  # 5,000 books per property
python scripts/compare_mechanisms.py                                 # pricing rules, side by side
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

## The battery: automatic by default, adjustable by agents

Surplus charges the battery before anything reaches the market, and a deficit draws
from it before anything is bought, so a household that does nothing trades only what
its battery could not absorb or supply. Since schema 0.2.0 an agent can nudge that
behaviour with `AgentDecision.battery_offset_kwh` — see **Battery control** below.

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

`clear_tick(tick, timestamp, orders, pricing=...)` takes a tick's `AgentDecision` book and
returns a `MarketState`. It is a **call auction**: every order for the half-hour arrives
together and the book is cleared once — not a continuous double auction, because there is
no order flow within a tick. Buys are grouped into price levels from the highest limit
down, sells from the lowest up, and the best levels trade while the buyers' maximum is at
least the sellers' minimum. Unmatched orders — including a partial fill's remainder — come
back in `MarketState` for the caller to settle against grid tariffs. The tick's
`clearing_price_eur_per_kwh` is the volume-weighted average of its trades, or `None` when
nothing cleared. Clearing is deterministic, and independent of the order the book arrives in.

## Ties are shared pro-rata

Orders at exactly the same limit price form one price level. When two crossing levels
trade, the side with more on offer shares the volume **in proportion to each order's
size**: two 1 kWh sellers facing a 1 kWh buyer sell 0.5 kWh each; a 3 kWh and a 1 kWh
seller sharing 2 kWh sell 1.5 and 0.5.

It used to be first-in-the-list-wins. Within a tick there is no arrival time, so "first"
was only the order the caller looped over households in — the first household in that
list won every tie, every tick, for a whole run. That mattered more than it sounds: the
rule-based baseline gives identical households identical prices, so ties are the normal
case. Pro-rata is the standard exchange rule for simultaneous orders, needs no seed, and
books without ties clear exactly as before.

## Pricing rules: `PricingRule`

Matching — who trades with whom, and how much — is identical under both rules; only the
price differs.

| Rule | Price | Property |
|---|---|---|
| `PAIRWISE_MIDPOINT` (default) | each pair at the midpoint of its own two limits | a buyer who bids high pays more; neighbours can pay different prices in one tick |
| `UNIFORM` | every trade at the midpoint of the marginal (last-matched) pair | one price per tick, inside every matched order's limit |

Choose with `MarketSimulator(households, pricing=PricingRule.UNIFORM)`.

**What changing it does** (`python scripts/compare_mechanisms.py`, rule-based baseline,
7 synthetic days):

| Households | Rule | Community savings | Buyers' share | Sellers' share | Price spread in a tick |
|---:|---|---:|---:|---:|---:|
| 4 | pairwise midpoint | €1.61 | 71.3% | 28.7% | 0.01 c/kWh |
| 4 | uniform | €1.61 | 71.3% | 28.7% | 0 |
| 20 | pairwise midpoint | €10.00 | 70.9% | 29.1% | 0.16 c/kWh |
| 20 | uniform | €10.00 | 70.4% | 29.6% | 0 |
| 100 | pairwise midpoint | €43.12 | 78.5% | 21.5% | 0.24 c/kWh |
| 100 | uniform | €43.12 | 77.6% | 22.4% | 0 |

**Community savings are identical, by construction:** what one neighbour pays for a kWh,
another receives, so with the same trades the price can only move savings *between*
households. The rule changes who saves — uniform pricing shifts about a point of the
saving from buyers to sellers at 100 households — and whether neighbours pay the same
for the same half-hour. The only way it can change the *total* is behaviour: a strategy
trained under one rule may bid differently under the other. That is the open experiment
to run with the RL policies.

## Property-based tests

`test_market_properties.py` generates random books with Hypothesis — up to 14
households, prices on a coarse grid so ties are constant — and checks, under both
pricing rules, that no book can break these invariants: traded plus unmatched volume
equals what each order offered; nobody trades outside their own limit; nothing that
could still cross is left on the book; shuffling the book changes no trade; tied orders
fill the same fraction; both rules agree on volume; and honest orders priced inside the
tariff gap never lose money at the meter. 200 books per property run in CI; the
`thorough` profile runs 5,000.
