# Agent Guidance for Codex

## Style
- Format with **black**; lint with **ruff**.
- Type-hint all public functions.

## Branch & PR rules
- Work on `dev/*` branches; open PRs into `main`.
- CI must pass: `ruff`, `pytest`, coverage ≥ 90 %.

## Tests
```bash
pytest -q
