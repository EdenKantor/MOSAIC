# MOSAIC

An experimental platform for studying whether temporary teacher access can become durable,
validated procedural capability under a fixed lifetime inference budget.

**Current scope: Milestone 0.** One agent, one task, one seed, JSONL events and raw usage accounting.
The deterministic provider is a scripted test fixture, not an LLM. This milestone establishes
experimental infrastructure; it provides no evidence for the learning hypothesis yet.

## Run the deterministic experiment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12 or newer.
From this repository:

```sh
uv sync --locked
uv run --locked python -m mosaic.experiments.run --config configs/pilot.yaml
```

The seeded `repair_pump` task requires fetching a spare part and returning to the pump.
Seed 42 requires 16 actions; seed 43 requires 6. The environment verifies the repair.
The command prints paths to `events.jsonl` and `summary.json` under `runs/fake-pilot/<run-id>/`.
Use `--output-dir runs/my-pilot` to choose a fresh directory. Existing directories are refused.

Exit codes: `0` environment-verified success, `1` completed unsuccessful task, `2` invalid setup
or runtime error. Runtime failures retain a trace and summary when the recorder remains writable.

## Check the implementation

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy mosaic tests
uv run --locked pytest -q
```

The default suite runs offline without Alem, Ollama, credentials, or external model calls.
The Alem test is skipped unless explicitly enabled. On Windows systems that block the compiled
mypy extension, install its source distribution into the same environment:

```sh
uv pip install --python .venv/Scripts/python.exe --no-binary mypy --reinstall-package mypy mypy==1.20.2
```

## Optional Alem CPU smoke

Alem is pinned to `14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e` (version 0.2.1), with upstream JAX
0.4.38. This extra requires **Python 3.12**; the deterministic core supports Python 3.12+.

```sh
uv sync --locked --extra alem --python 3.12
uv run --locked --extra alem python -m mosaic.experiments.run --config configs/alem-smoke.yaml
```

The smoke uses one player, coordination disabled, and a scripted `Noop` action. Its goal is
`collect_wood`, so its expected result is **failure, exit 1**. It verifies simulator wiring,
not model capability. The initial JAX compilation can take several minutes on CPU.
To run the optional integration test in PowerShell:

```powershell
$env:MOSAIC_TEST_ALEM = "1"
$env:JAX_PLATFORM_NAME = "cpu"
uv run --locked --extra alem pytest -q -m alem
```

See [Alem integration notes](docs/alem.md), [architecture](docs/architecture.md),
[experiment semantics](docs/experiments.md), and [research scope](docs/research.md).
