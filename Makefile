.PHONY: setup lint test run-sim ai-context clean

# Override on any machine that doesn't have 3.12: `make setup PYTHON=python3.11`
PYTHON ?= python3.12

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev]"
	@echo "Run 'source .venv/bin/activate' to activate the environment."

lint:
	. .venv/bin/activate && ruff check .

test:
	. .venv/bin/activate && pytest

run-sim:
	. .venv/bin/activate && python -m dashboard.orchestrator

ai-context:
	@for dir in . agents dashboard forecasting simulation; do \
		ln -sf AGENTS.md $$dir/CLAUDE.md; \
	done
	@echo "Linked CLAUDE.md -> AGENTS.md (gitignored; Claude Code reads CLAUDE.md only)"

clean:
	rm -rf .venv .pytest_cache .ruff_cache **/__pycache__
