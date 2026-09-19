# CLAUDE.md — simulation/

Read the root `CLAUDE.md` first — this file only covers what's specific to this module.

## Scope

This module owns:
1. The **household environment** — each household's state each tick (current demand, current solar generation, current battery level), derived from `HouseholdProfile` plus real/synthetic time-series data.
2. The **market clearing mechanism** — a double auction cleared once per tick (a call auction — every order for a tick arrives together, so there is no continuous order flow and no time priority): collect `AgentDecision` orders (buys and sells) for a tick, match them by price priority, produce `TradeEvent`s for matched orders and leave the rest as unmatched orders in `MarketState`.

3. The **tick loop** — `simulator.MarketSimulator.step()`, which runs the physics, clears the book and settles the tick against each household's metered net position, in that order. This is the module's public entry point: `agents/` and `dashboard/` call it and nothing else. Keep the order of those three steps in one place; two callers doing it by hand is two chances to disagree and produce two different headline numbers.

This module does **not** decide what any household bids — that's `agents/`. It only clears whatever orders it's given.

The battery is environment-owned physics, not an agent decision (team default, see this folder's `README.md`). Don't move charge/discharge into `AgentDecision` without the user confirming the team agreed to it — it changes the RL action space.

## Inputs / outputs (contracts, not internals)

- **Consumes:** `AgentDecision` (from `agents/`, one per household per tick)
- **Produces:** `TradeEvent` and `MarketState` (for `dashboard/` and `agents/`)

## Suggested build order

1. Start with a tiny deterministic test case: 2 households, fixed known bids, verify the auction clears exactly as expected by hand-calculation. This is your contract test — write it before the general mechanism.
2. Household environment: load a real household's half-hourly consumption from `data/`, and a solar generation curve (from PVGIS or the synthetic fallback), and expose per-tick net position.
3. General N-household double auction. Price priority is the standard, well-understood approach — don't reach for something more exotic unless the simple version genuinely doesn't work for your case. Within a tick there is no arrival time to prioritise by, so never let an order's position in the input list decide who trades.
4. Handle the "no trades cleared" case cleanly — `MarketState.clearing_price_eur_per_kwh` is `None`, unmatched orders fall back to grid import/export tariffs from `HouseholdProfile`. This fallback path is what makes the P2P-vs-grid-baseline comparison in the dashboard meaningful.

## Testing expectations

- All randomness (household assignment, any stochastic elements) must accept a seed — reproducibility matters when the RL team is trying to debug why an agent behaved unexpectedly on a specific run.
- The 2-household hand-calculated test case above should stay in the test suite permanently as a sanity check — if it ever breaks, something fundamental changed.
