# MOSAIC — Milestone 1

**Routing Only, baseline B3:** bounded weak execution and recovery, then teacher takeover of the
remaining episode. Nothing learned from the teacher persists into another episode.

This folder is independently runnable. Milestone 0 is preserved in `../milestone-0/`.
Both model policies in these examples are deterministic fixtures; they are not real LLMs.

## Deterministic routing experiment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use Python 3.12+.
Run from this milestone folder:

```sh
uv sync --locked
uv run --locked python -m mosaic.experiments.run --config configs/routing.yaml
```

Seed 42 produces **2 weak calls, 16 teacher calls, 18 environment actions**, and an
environment-verified pump repair. One weak turn and one recovery turn choose an advertised
wait; both finite success windows expire. The teacher then performs the existing 16-action
fetch-and-repair sequence. No new simulator was introduced.

Additional configurations:

| Configuration | Expected outcome |
| --- | --- |
| `configs/weak-success.yaml` | Success, 16 weak calls, 0 teacher calls |
| `configs/recovery-success.yaml` | Success in recovery, 17 weak calls, 0 teacher calls |
| `configs/teacher-withdrawn.yaml` | Blocked escalation, 2 weak calls, 0 teacher calls, failure |
| `configs/pilot.yaml` | Original deterministic weak-only experiment |

Each command writes `events.jsonl` and `summary.json` under `runs/<experiment-id>/<run-id>/`.
Use `--output-dir runs/my-routing` to select a fresh directory. Existing directories are refused.
Exit codes are 0 success, 1 completed unsuccessful episode, and 2 setup/runtime error.

## Routing and accounting

The policy uses consecutive execution failures or an exhausted decision window without verified
goal success. It allows the configured bounded weak recovery before escalation. Valid multi-step
actions do not each trigger escalation. Neither confidence nor requests for help control routing.

Teacher inference is a separate, revocable runtime capability. Withdrawn access drops its provider
reference and blocks invocation before dispatch/accounting. Model requests contain visible
observations, task, available actions and bounded feedback from the current episode. Snapshots
and verifier evidence are excluded. Reusing a runner starts with empty history.

The trace labels every model call and action with its role. The budget contains `weak` and
`teacher` groups with calls, token totals, unknown usage, synthetic usage and model-call wall
time, plus aggregate usage and environment actions. Summaries include teacher-call fraction,
escalations and blocked escalations. Fake token counts are explicitly synthetic, not API costs.

See [exact routing semantics and scientific limitations](docs/routing.md),
[current architecture](docs/architecture.md), [experiments](docs/experiments.md), and
[research scope](docs/research.md).
The [validation report](docs/validation.md) records exact executed checks and their results.

## Checks

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked python -m mypy mosaic tests
uv run --locked python -m pytest -q
```

The core suite runs without Alem, Ollama, credentials or external model calls. It includes all
26 original Milestone 0 tests and the new routing invariants. Two optional Alem tests are skipped
unless enabled. On Windows systems blocking the compiled mypy extension, use its source build:

```sh
uv pip install --python .venv/Scripts/python.exe --no-binary mypy --reinstall-package mypy mypy==1.20.2
```

## Optional Alem routing smoke

Alem remains pinned to `14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e`, version 0.2.1, MIT.
Its extra requires Python 3.12 and uses the upstream JAX 0.4.38 CPU backend.

```sh
uv sync --locked --extra alem --python 3.12
uv run --locked --extra alem python -m mosaic.experiments.run --config configs/alem-routing-smoke.yaml
```

Both fixtures choose Noop. The result is expected failure (exit 1), with 2 weak calls, 1 teacher
call and 3 environment actions. This exercises routing against the real simulator; it is not
an intelligence test or a comparable Alem leaderboard result.

Run both optional tests in PowerShell:

```powershell
$env:MOSAIC_TEST_ALEM = "1"
$env:JAX_PLATFORM_NAME = "cpu"
uv run --locked --extra alem python -m pytest -q -m alem
```

[Alem integration notes](docs/alem.md) describe the adapter and source inspection.
No real model provider or large model weights are installed by this milestone.
