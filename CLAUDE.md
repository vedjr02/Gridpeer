# CLAUDE.md — Root Context

This file is read by every Claude Code session working in this repo, regardless of which of the four of you is running it. Keep it accurate — if the architecture changes, update this file in the same PR.

## What this project is

GridPeer is a simulated peer-to-peer energy trading marketplace. Households (agents) with solar + battery + variable demand trade energy directly with each other through a market-clearing mechanism, instead of just exporting surplus to the grid. The point of the project is the *comparison*: do learning agents produce better outcomes (household savings, CO2 avoided, peak-load reduction) than a rule-based baseline?

Full context: `README.md`. Build plan: `ROADMAP.md`.

## The one rule that matters most

**`shared/schemas.py` is the single source of truth for how modules talk to each other.** No module reaches into another module's internals — they only exchange the Pydantic models defined there (`TradeEvent`, `AgentDecision`, `ForecastOutput`, `HouseholdProfile`, `MarketState`, `HouseholdState`).

If you (the AI assistant) are working inside `simulation/`, `agents/`, `forecasting/`, or `dashboard/`, and you find yourself wanting to import something from a sibling module directly, stop — that almost always means the contract in `shared/` is missing a field, not that a direct import is the right fix. Flag it to the user instead of working around it.

**Never rename or remove a field in `shared/schemas.py` without an explicit instruction from the user that all four teammates have agreed to the change.** See `CONTRIBUTING.md` for the schema-change process. A silent breaking change here breaks someone else's module who isn't in this conversation.

## Conventions

- Python 3.11+, full type hints on every public function.
- All cross-module data is a Pydantic model from `shared/`. Never pass around bare dicts between modules.
- Formatting/linting: `ruff` (config in `pyproject.toml`). Run `make lint` before committing.
- Tests: `pytest`, one test file per module minimum, plus `tests/test_contracts.py` which every module's output must satisfy.
- Docstrings on every public class/function — one line of intent, not a restatement of the signature.
- Units are always explicit in variable/field names or docstrings (kWh, €/kWh, kW). Energy trading bugs are usually unit bugs.
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`) — see `CONTRIBUTING.md`.

## Module boundaries (read the module's own CLAUDE.md for detail)

- `simulation/` — market clearing mechanism + household environment. Consumes `AgentDecision` (including its `battery_offset_kwh`), produces `TradeEvent`, `MarketState` and `HouseholdState` (battery state, schema 0.2.0).
- `agents/` — trading strategies (baseline + RL). Consumes `ForecastOutput`, `MarketState` and, when batteries are on, `HouseholdState`; produces `AgentDecision`.
- `forecasting/` — demand + solar generation forecasting. Produces `ForecastOutput`.
- `dashboard/` — orchestration loop (ticks the simulation forward) + live visualisation. Consumes everything, persists `TradeEvent` history.

## Running things

```bash
make setup      # venv + install
make lint       # ruff check
make test       # pytest, all modules
make run-sim    # end-to-end run with whatever's implemented so far
```

## What "done" looks like for week 1

An ugly but fully wired pipeline: simulation emits `TradeEvent`s (even from dummy logic), agents produce `AgentDecision`s (even a hardcoded one), forecasting returns a `ForecastOutput` (even a flat line), and the dashboard renders something from real data flowing through real contracts — no module's output is hand-typed into another module's input. If you're asked to help build any one module before this pipeline exists, build the minimal version that satisfies the contract first, then improve it.
