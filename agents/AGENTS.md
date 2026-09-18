# AGENTS.md — agents/

Read the root `AGENTS.md` first — this file only covers what's specific to this module.

## Scope

This module decides what each household bids or asks each tick. It owns two strategy families:

1. **Baseline (build this first, non-negotiable):** a rule-based or simple bandit strategy — e.g. bid/ask at a price partway between the grid import and export tariff, adjusted by forecasted net position. This is not a throwaway — it is the yardstick every RL result gets compared against for the rest of the project. Get it running through the real pipeline before touching RL.
2. **RL agents:** multi-agent reinforcement learning, one policy per `AgentRole` (or per household, if compute allows). Learns a bidding/asking strategy that should beat the baseline on household savings.

## Inputs / outputs (contracts, not internals)

- **Consumes:** `ForecastOutput` (per household, from `forecasting/`), `MarketState` (previous tick's outcome, from `simulation/`)
- **Produces:** `AgentDecision` (one per household per tick, consumed by `simulation/`)

## Suggested build order

1. Baseline strategy, wired through the real pipeline end to end. This unblocks the dashboard owner (they now have a real comparison point) and gives you a sanity-checked environment before RL adds its own instability on top.
2. Set up the RL environment using PettingZoo's API conventions even if you're not using PettingZoo's training loop directly — it's the de facto standard for multi-agent RL environments and makes your environment legible to anyone who's used the ecosystem before.
3. Start training with Stable-Baselines3 (PPO is a reasonable default for a first pass). Don't reach for RLlib unless SB3 genuinely can't handle what you need — the setup cost isn't worth it for a first working result.
4. Reward function: the natural first cut is household savings vs. the grid-baseline fallback for that tick, possibly with a small penalty for unmatched (failed) orders so agents don't learn to bid unrealistically.

## Evaluation harness

This is arguably the most important deliverable of this module, more than the RL training itself: a script that runs a full simulation with the baseline strategy, then again with the trained RL policy, and outputs the comparison the whole project is built to produce (`RunSummary` for each). Build this early — even against the skeleton pipeline in week 1 — so you're not improvising it under deadline pressure in week 3.

## A note on scope discipline

Multi-agent RL is genuinely hard to get training stably. If by mid-week-3 the RL agent isn't clearly beating the baseline, that's not a failure — a well-reasoned "here's the baseline, here's what we tried with RL, here's why it's harder than it looks" is a legitimate and honest result. Don't let RL instability block the rest of the team's integration work; keep the baseline as the default strategy flowing through the pipeline at all times.
