# Milestone 2 validation report

Validation date: 2026-10-02. Windows, Python 3.12.14. The core and optional Alem checks
used separate editable installations with locked dependencies. No new project dependencies,
real model calls or model-weight downloads were required.

## Implemented

Teacher-assisted acquisition, deterministic candidate compilation, fresh weak-only validation,
atomic promotion to a shared JSON PlayBook and persisted reuse by a new agent. Each candidate
records scope, declared visible initiation conditions, bounded procedure, allowed tools,
termination goal, source provenance and content identity. Schema-3 lifecycle events and
phase-separated accounting make acquisition and memory overhead inspectable.

Milestone 2 is independently runnable in `milestones/milestone-2`. The existing
`milestones/milestone-0` and `milestones/milestone-1` source, tests and lockfiles are preserved.

## Architecture

The pipeline composes the original episode runner with `mosaic/skills/`. The compiler accepts
a consistent, verified successful source trace containing accepted teacher actions, then
receives only visible observations/actions/feedback and the success gate. Snapshots and
private verifier evidence are excluded from the compiler's input and from model requests.

Validation constructs a fresh environment, weak provider and runner for every predetermined
seed. Neither a teacher capability nor a routing policy is supplied. Every trial must execute
the complete candidate and pass the environment verifier before promotion. Missing initiation,
divergence, rejection, partial execution or unverified termination prevents promotion.

The PlayBook checks schema, content hashes and validation consistency on reopen. Retrieval
matches environment, pinned revision, task and declared visible conditions. A selected procedure
is supplied as a hint to ordinary weak requests; each action still incurs inference, parsing,
environment dispatch and verification. A miss continues ordinary execution. A failed execution
removes its hint and records the failure; validation stops immediately.

## Tests run

From `C:\Users\ngede\Documents\Codex\2026-10-02\g\outputs\MOSAIC\milestones\milestone-2`,
the core verification commands were:

```powershell
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\milestone-2-venv\Scripts\python.exe' -m ruff format --check mosaic tests
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\milestone-2-venv\Scripts\python.exe' -m ruff check mosaic tests
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\milestone-2-venv\Scripts\python.exe' -m mypy mosaic tests
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\milestone-2-venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider --basetemp 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\pytest-m2-final-20261002c'
```

Results: **137 passed, 2 skipped in 21.10 seconds**; lint passed; 38 files already formatted;
strict type checking passed for 38 files. The optional Alem tests are the two skipped checks.

The lockfile check also passed, resolving the same 59 locked packages offline:

```powershell
$env:UV_CACHE_DIR='C:\Users\ngede\Documents\Codex\2026-10-02\g\work\uv-cache'
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\tooling-m2\bin\uv.exe' lock --check --offline --python 'C:\Users\ngede\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
```

Coverage includes retained routing/withdrawal invariants, compiler evidence and identity
checks, private-state canaries, bounded procedures, candidate/report integrity, duplicate
promotion, atomic persistence failure, all-seed validation, refusal of divergent/partial/illegal
procedures, context exclusions, teacher-free reuse, CLI outcomes, deterministic reuse of a
fixed PlayBook, exact prompt bytes and reconciliation against raw call/action events.
Successful but inconsistent teacher traces fail compilation. Unknown usage stays unknown.
Acquisition/validation/reuse transport failures retain prior costs and the failed call's unknown
usage. Later declared validation/reuse episodes still run. A corrupted PlayBook records its
failed retrieval attempt and overhead without dispatching inference or overwriting evidence.

The preserved Milestone 1 suite was rerun before creating this folder: **41 passed, 2 skipped
in 7.23 seconds**. No earlier milestone files changed.

Optional real Alem checks used the pinned dependency in a separate environment:

```powershell
$env:MOSAIC_TEST_ALEM='1'
$env:JAX_PLATFORM_NAME='cpu'
$env:MPLCONFIGDIR='C:\Users\ngede\Documents\Codex\2026-10-02\g\work\m2-matplotlib'
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\milestone-2-alem-venv\Scripts\python.exe' -m pytest -q -m alem -p no:cacheprovider --basetemp 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\pytest-m2-alem-20261002a'
```

Result: **2 passed, 131 deselected in 128.29 seconds**. Collection preceded the last compiler
cases; those cases are covered by the final core suite. Fresh workspace temporary directories
avoided the host's inaccessible default temporary test directory.

Portable checks after `uv sync --locked` are `uv run --locked python -m pytest -q`,
`uv run --locked ruff check mosaic tests`, `uv run --locked ruff format --check mosaic tests`
and `uv run --locked python -m mypy mosaic tests`. Alem requires
`uv sync --locked --extra alem --python 3.12` and the environment variables above.

## Example run

From the Milestone 2 directory:

```sh
uv run --locked python -m mosaic.experiments.learn --config configs/skill-pilot.yaml --output-dir runs/skill-pilot
uv run --locked python -m mosaic.experiments.run --config configs/skill-reuse.yaml --output-dir runs/independent-skill-reuse
```

The examples were executed with the installed interpreter as follows:

```powershell
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\milestone-2-venv\Scripts\python.exe' -m mosaic.experiments.learn --config configs/skill-pilot.yaml --output-dir runs/skill-pilot
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\milestone-2-venv\Scripts\python.exe' -m mosaic.experiments.run --config configs/skill-reuse.yaml --output-dir runs/independent-skill-reuse
```

The scripted seed-42 pilot acquires a 16-action pump procedure,
`skill-38cb7756ffe8c336`, validates it, promotes it and reopens the PlayBook for reuse by
`agent_1`. The independent reuse configuration uses `agent_2`.

| Phase | Weak calls | Teacher calls | Environment actions |
| --- | ---: | ---: | ---: |
| Acquisition | 2 | 16 | 18 |
| Validation | 16 | 0 | 16 |
| Pipeline reuse | 16 | 0 | 16 |
| Pipeline total | **34** | **16** | **50** |
| Additional independent reuse | 16 | 0 | 16 |

The pipeline counts one deterministic compilation, zero compiler-model calls and one local
retrieval. Each validation/reuse action still requires a weak call. It records compilation and
retrieval time, exact injected UTF-8 skill bytes, full-request synthetic usage, separate phase
budgets and total pipeline wall time. The additional independent episode has its own ledger;
it is not silently included in the pipeline total.

Changing the validation seeds to `[42, 43]` runs both trials and refuses promotion when the
different visible part location does not match. The successful seed-42 trial's 16 weak calls
remain counted. This is refusal of unsupported conditions, not a transfer success.

## Trace

`runs/skill-pilot/events.jsonl` records learning start, compilation, candidate creation,
validation, promotion and finish. Its `summary.json` reconciles all phases. Full inference and
action evidence remains in `acquisition/events.jsonl`, `validation/42/events.jsonl` and
`reuse/42/events.jsonl`. Candidate, validation report and PlayBook are separate JSON artifacts.

The reuse trace records local retrieval and execution start, 16 role-tagged model calls and
normal action dispatches, completed procedure, and environment-verified success. The source
manifest records commit/dirty state, code/configuration/lockfile fingerprints and runtime
identities. Existing output directories are refused to preserve earlier evidence.

## Alem status

The unchanged adapter uses official Alem 0.2.1 at commit
`14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e`. The checks ran its original single-Noop
smoke and two-weak/one-teacher routing smoke on Windows CPU with JAX 0.4.38.
Expected environment-verified failure confirms integration; it does not establish task
intelligence. No skill-acquisition result is claimed for Alem's text observations. The initial
compiler requires declared fields from visible JSON; an Alem-compatible skill representation
would require separate design and validation.

## Decisions and tradeoffs

Compilation extracts a finite demonstrated sequence without another model call. Exact visible
conditions and all-pass validation favor refusal over coverage. There are no branches, loops,
parameter binding, composition, autonomous repair or learned investment rules. The initial
pilot repeats seen conditions with stateless fake providers and another agent ID. It proves
persistence, validation gates and accounting, not language-model learning or net savings.

Stored reports are trusted experiment artifacts. Schema checks and content hashes are
consistency checks, not authentication of edited provenance, traces or validation results.
The PlayBook supports one writer. Filesystem failures may interrupt evidence recording; the
pipeline marks `budget_complete=false` when a phase cannot return complete artifacts, while
retaining completed prior phases. Such a run cannot support complete-cost comparisons.

Acquisition, validation, compilation, retrieval and repeated hint overhead must all be included
in later comparisons. Synthetic whitespace token counts do not estimate API cost. The decision
cap does not enforce a lifetime token budget. No real-model adapter, comparative baseline study,
near transfer or global teacher-withdrawal experiment is claimed.
Local Ollama was checked again: its command and local service were unavailable.

## Next recommended step

**Milestone 3 — Hard Teacher Withdrawal**: implement a predefined irreversible teacher boundary
across experiment services and evaluate retained PlayBook capability with the full pre/post
cost ledger. Milestone 3 has not been implemented.
