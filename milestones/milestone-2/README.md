# MOSAIC — Milestone 2

**Teacher → Candidate Skill → Weak-Agent Validation → Shared PlayBook.**

This folder adds procedure validation gates and persistent shared memory to the Milestone 1
runner. Its **engineering scaffold is complete; it is not scientifically validated**.
The deterministic fixtures demonstrate plumbing, validation gates, persistence and accounting.
They do not demonstrate useful transfer, real-model learning or economic savings. Milestones
0 and 1 remain independently runnable in their own folders.

## Run acquisition and reuse

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use Python 3.12+.
Run from this milestone folder:

```sh
uv sync --locked
uv run --locked python -m mosaic.experiments.learn --config configs/skill-pilot.yaml --output-dir runs/skill-pilot
```

The pipeline runs bounded weak execution and recovery, then teacher takeover. Verified
teacher success produces a bounded declarative candidate from accepted teacher actions.
Fresh weak-only episodes validate it on the configured seeds. Promotion requires every
validation episode to follow the complete procedure and achieve environment-verified success.
A JSON PlayBook persists only promoted procedures and their validation reports.

The default demonstration acquires on seed 42, validates on seed 19 and reuses on seed 20.
All three have initial part location 7 and matching selected visible starting conditions.
Reuse uses another agent ID and a newly opened PlayBook. Independent seeds here demonstrate
**seen-condition reuse**, not held-out generalization, useful near transfer or composition.

After that command creates the PlayBook, run an independent reuse episode:

```sh
uv run --locked python -m mosaic.experiments.run --config configs/skill-reuse.yaml
```

Each weak action still requires a model call. The request contains the selected procedure
and its cursor; the runner checks the action through its usual parser and environment
interface. The fake `follow_skill` policy follows the supplied hint and otherwise waits.
A retrieval miss continues ordinary weak execution. Procedure divergence or rejection records
a skill failure; ordinary execution may continue on the next turn.

The compiler currently stores only the teacher suffix and activates it under the teacher's
takeover conditions. Retrieval happens only at episode initialization, and validation also
starts from reset. The fixture's weak prefix waits and changes only excluded step metadata,
so those conditions happen to match. If a real weak agent makes progress before takeover,
the suffix may not apply at reset. This unresolved semantic mismatch blocks real-trajectory
skill claims. Calibration comes first; [skill semantics](docs/skills.md) records two redesign
options without implementing either.

## Traces and persistence

The pipeline writes these paths under `runs/skill-pilot/`:

| Path | Contents |
| --- | --- |
| `events.jsonl`, `summary.json` | Pipeline lifecycle, outcomes and accounting |
| `acquisition/` | Full acquisition episode trace and summary |
| `candidate.json` | Candidate procedure with scope, conditions and provenance |
| `validation/<seed>/`, `validation.json` | Independent weak-only validation traces and report |
| `playbook.json` | Integrity-checked promoted procedure and validation evidence |
| `reuse/<seed>/` | Weak-only reuse traces and summaries |

Output directories must be fresh. Exact results and executed checks belong in
[the validation report](docs/validation.md).

Retrieval requires matching environment, pinned revision, task and configured visible
conditions. The pilot matches `position`, `part_position`, `has_part` and `pump_repaired`.
A different part location is refused. The procedure is an action sequence, not generated
Python. There are no embeddings or teacher-assisted retrieval.

Compilation and retrieval record local operation counts and wall time. Reports keep
acquisition, validation and reuse inference separate, including weak/teacher calls, unknown
usage, synthetic usage and environment actions. Exact UTF-8 injected skill bytes are recorded;
provider input usage includes the whole request. Fake token counts are synthetic, not API costs.

See [skill semantics](docs/skills.md), [architecture](docs/architecture.md),
[accounting](docs/experiments.md) and [research limits](docs/research.md).
Routing-only configurations remain available without skills.

## Checks

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked python -m mypy mosaic tests
uv run --locked python -m pytest -q
```

Core CI covers the independently runnable M0/M1/M2 folders on Python 3.12 without Alem or
ordinary real-provider calls. The core suite needs no Alem installation, credentials or
real-model service. On Windows
systems blocking the compiled mypy extension, use its source build:

```sh
uv pip install --python .venv/Scripts/python.exe --no-binary mypy --reinstall-package mypy mypy==1.20.2
```

## Alem and real models

The existing Alem adapter remains pinned to
`14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e`, package 0.2.1, MIT. Its Python 3.12 extra
preserves the original single-agent and routing smoke configurations:

```sh
uv sync --locked --extra alem --python 3.12
uv run --locked --extra alem python -m mosaic.experiments.run --config configs/alem-routing-smoke.yaml
```

That scripted Noop smoke expects goal failure (exit 1). It exercises integration rather
than Alem capability. This milestone does not claim meaningful skill acquisition for
Alem's text observations; see [Alem notes](docs/alem.md). No real-model provider or large
weights are added.

The tested `TeacherCapability.withdraw()` remains a runtime primitive; no scheduled M3 study
is implemented. The revised sequence is measurement/provider hardening (M1.25), real-model
calibration (M1.5), protocol freeze (M1.75), real-trajectory skill semantics (M2-R), hard teacher
withdrawal (M3), then economics (M4).

The next recommended step is to **run real-model calibration** after provider and measurement
hardening.
