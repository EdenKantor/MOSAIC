# Milestone 1.25 verification and calibration status

## Implemented

This folder adds Ollama/Groq/Gemini provider adapters, five raw usage dimensions with request/source
metadata, and an official solo Alem calibration prompt. The adapters are **implemented but
uncalibrated**. Mocked tests and scripted fixtures do not establish useful model performance,
useful transfer or savings. No M3 experiment or skill redesign is included.

## Architecture

One provider invocation uses one standard-library non-streaming HTTP POST. There are no SDK
retries, redirect following, parser repair calls or fallback models. The runner records every
attempt and keeps missing usage unknown. Actor requests contain reviewed rules, visible task
state, legal solo actions and bounded visible history; snapshots and verifier evidence remain
audit data. Secrets come only from the environment.

Accounting preserves input, output, reasoning, cached-input and provider-total dimensions,
known subtotals and unknown-call counts in aggregate and by role. Reasoning/cache dimensions
are not added again to reported totals. See [provider semantics](providers.md).

The new Alem calibration adapter uses the pinned official solo wrapper, level-gated rules,
precise public coordinates and consistent solo affordances. Soft specialization with both
role efficiencies 1.0 makes the registered resource/tool tasks reachable for the solo warrior.
These are explicit conditions for both model roles, not the official cooperative benchmark.
See [the source comparison](alem-prompt-comparison.md).

## Current verification — 2026-10-03

Locked dependency synchronization and Python 3.12 formatting, lint, strict typing and offline
tests passed before live smoke. Ordinary CI excludes the optional Alem marker and installs no
Alem/JAX or provider SDK. The final M1.25 rerun followed the model-specific 404 correction
and the fix for generic billing links in HTTP 429 error bodies.

| Folder | Current result | Test time | Type-checked files |
| --- | --- | --- | --- |
| M0 | 26 passed; 1 Alem test deselected | 5.43 s | 23 |
| M1 | 41 passed; 2 Alem tests deselected | 8.54 s | 26 |
| M2 | 195 passed; 2 Alem tests deselected | 27.27 s | 41 |
| M1.25 | 265 passed; 3 Alem tests deselected | 28.70 s | 39 |
| Laboratory | 16 stdlib tests passed | 1.122 s | 6 |

The optional pinned-Alem crafting/rule protocol check also passed: 1 test, 10 deselected,
27.97 seconds, with zero model calls. Laboratory JavaScript syntax and replay logic passed;
browser visual inspection was unavailable. The source-build mypy workaround remains local
to this Windows host; ordinary Linux CI uses its locked dependency installation.

New mocked coverage includes Gemini normalization/usage, captured/current secret redaction,
quota headers, single physical attempts, 13+ registered candidates, prior-batch hydration,
model-specific 404 exclusion, request/token guards before call reservation, provider-fatal
stops, incomplete-ledger stops and infrastructure exclusions from capability denominators.
The existing withdrawn-teacher boundary remains checked before a provider adapter dispatch.

Real free-only access/smoke/diagnostic evidence is separate from these tests. Its current
status is recorded under [M1.5 reports](../../milestone-1.5/reports/2026-10-03/).
No useful capability gap, transfer or withdrawal claim follows from an infrastructure smoke.

## Historical tests — 2026-10-02

Local verification completed on Python 3.12. These portable locked commands express the checks
run from each independently runnable package:

```sh
uv run --locked ruff format --check mosaic tests
uv run --locked ruff check mosaic tests
uv run --locked python -m mypy mosaic tests
uv run --locked python -m pytest -q -p no:cacheprovider
```

| Folder | Core pytest result | Time | Type-checked files | Formatting / lint |
| --- | --- | --- | --- | --- |
| M0 | 26 passed, 1 optional skipped | 5.44 s | 23 | Passed |
| M1 | 41 passed, 2 optional skipped | 8.71 s | 26 | Passed |
| M2 | 195 passed, 2 optional skipped | 21.64 s | 41 | Passed |
| M1.25 | 167 passed, 3 optional skipped | 13.47 s | 36 | Passed |

The local run used pre-synced isolated environments, a source build of mypy
(`MYPY_USE_MYPYC=0`), workspace `TEMP`/`TMP`, and fresh workspace pytest basetemp directories.
M1.25 pytest additionally used
`--basetemp work/pytest-m125-hardening-final-20261002b` under the session workspace.
Locked dependency synchronization passed offline from the populated cache for all four core
packages: 59 packages resolved and 19 installed per core environment. The pinned M1.25 Alem
extra installed 59 packages. Offline installation assumes that dependency cache is populated.

Ordinary core tests use fake model policies and injected mocked HTTP transports. They require
no Alem, credentials or real-model service. Optional pinned-Alem checks use the real simulator
without inference. Real-model calibration remains separate from both sets of checks.

The optional protocol check used the pinned Alem environment with `MOSAIC_TEST_ALEM=1`,
`JAX_PLATFORM_NAME=cpu` and a workspace `MPLCONFIGDIR`:

```sh
uv run --locked --extra alem python -m pytest -q tests/test_calibration_protocol.py -m alem -p no:cacheprovider
```

Result: **1 passed in 18.96 s**, with zero model calls. It checks the grounded solo prompt,
registered achievement goals and crafting reachability under controlled test states/materials.
This is simulator/protocol evidence, not a successful real-model trajectory or benchmark score.
These are local verification results; hosted CI/publication status is reported separately.

## Historical preparation example — 2026-10-02

The earlier calibration procedure was configured, not an empirically selected pair.
The following records its original non-inference check; current real runs must use the
candidate-selection gates and free-only registrations described in M1.5.

```sh
uv sync --locked --extra alem --python 3.12
uv run --locked --extra alem python -m mosaic.experiments.run --calibration-config ../milestone-1.5/configs/calibration-pilot.yaml --preflight
uv run --locked --extra alem python -m mosaic.experiments.run --calibration-config ../milestone-1.5/configs/calibration-pilot.yaml --output outputs/calibration-pilot
```

The actual non-inference preflight reported:

| Check | Result |
| --- | --- |
| Pinned Alem source | Passed |
| Local Ollama service / configured model | Unavailable |
| `GROQ_API_KEY` in the environment | Missing |

That historical preflight was not ready. It made zero inference calls and zero Groq network
requests; no real pilot was run then. Current API-only prerequisites passed on 2026-10-03
after credentials became available. Credential presence alone never establishes remote access.

The pilot independently evaluates weak-only and teacher-only
arms for four task families across seeds 42, 19, 20, 7 and 11. Both have the same rules, visible
history bound of eight, temperature 0 and output cap 2,048. Ollama uses `qwen3:4b`, thinking on,
and context window 32,768; Groq uses `openai/gpt-oss-120b` and reasoning effort `medium`.
Those deliberation controls differ deliberately and are recorded rather than treated as
identical mechanisms.

Review the pilot before invoking the 20-seed expansion with `--confirm-pilot-reviewed`.
Configured model names are not claims of current service availability or a measured weak /
teacher capability gap.

## Trace

A calibration run writes these paths under the selected fresh output directory:

| Path | Evidence |
| --- | --- |
| `config.json` | Exact frozen calibration configuration |
| `summary.json`, `report.md` | Paired outcomes, raw costs and family/role aggregates |
| `<family>/<seed>/<role>/arm.json` | Preparation, initial hashes, outcome and accounting |
| `<family>/<seed>/<role>/events.jsonl`, `summary.json` | Full episode evidence when setup reaches execution |

Initial snapshot, advertised actions and visible prompt hashes are compared before inference;
recorded episode reset evidence must match them too. Incomparable pairs remain recorded and
excluded from comparable outcome aggregates; their returned costs remain in total raw costs.
Preserve configurations and the committed runtime alongside traces. Invalid JSON, rejected
actions, provider failures and missing usage remain evidence. No real calibration trace or
favorable score is fabricated here. CLI exit 0 means the protocol completed, including valid
goal failures; exit 2 marks setup/runtime or comparability errors.

## Alem status

Alem remains pinned to `14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e`, package 0.2.1.
Success for each focused task reads its explicit official achievement bit. The new prompt
fixes the earlier minimal real-model interface and the solo hard-role task mismatch; source
inspection and simulator checks are not model capability evidence. Earlier M0–M2 adapters
remain unchanged.

## Decisions and tradeoffs

Calibration has no skills, PlayBook, teacher escalation or withdrawal intervention. It first
asks whether the chosen weak/teacher models behave usefully under a grounded common interface.
The setting differs from upstream in player count, coordination, efficiencies, focused-task
outcomes, strict JSON output, history serialization, reasoning controls and retry protocol.
Neither benchmark equivalence nor lifetime-budget savings is claimed.

M2 remains an engineering scaffold with an unresolved teacher-suffix/reset activation mismatch.
Measurement hardening precedes real-model calibration, protocol freeze, M2-R semantics, M3 hard
withdrawal and M4 economics. The tested legacy withdrawal capability remains a primitive.

## Next recommended step

Resume real-model candidate calibration within verified available free quota, select a pair
from empirical evidence, and register its five-seed pilot before skill/withdrawal claims.
