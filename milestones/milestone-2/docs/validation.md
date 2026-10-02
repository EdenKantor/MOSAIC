# Milestone 2 engineering validation report

Validation date: 2026-10-02. Windows, Python 3.12.14. Earlier milestone folders and the
tested teacher-withdrawal primitive are preserved. No additional M3 functionality or M3
research run is included in this correction.

Milestone 2 demonstrates plumbing, validation gates, persistence and accounting with
deterministic fixtures. It does not demonstrate useful procedural transfer between real models.
It is a B5 procedural-memory engineering scaffold, complete but scientifically unvalidated.

## Scope and architectural blocker

Acquisition, bounded deterministic compilation, weak-only validation, atomic PlayBook
promotion and fresh-agent reuse remain implemented. Compilation receives visible
observations, accepted actions and feedback, plus the verified-success gate. Private audit
snapshots and verifier evidence do not enter the compiler or actor requests.

**BLOCKER for real procedural-memory experiments:** compilation extracts the accepted
teacher trajectory, whose initiation observation is the takeover state. Retrieval and
validation activate from task reset. Fake weak actions are waits, and the selected initiation
fields exclude the step counter, so the visible conditions coincide in this pilot. Real weak
progress generally breaks that assumption. The implementation and its predicates are not
silently weakened or redesigned.

Before M2-R, choose either full accepted reset-to-success workflow skills, with reset-state
initiation and validation, or teacher-suffix/subgoal skills with dynamic activation and
scientifically defensible reachable activation-state validation. Full-task workflows are
preferred for Experiment #1 unless real calibration trajectories provide a strong reason
otherwise. See [the detailed alternatives](skills.md).

## Current checks

Core checks after the seed/accounting changes:

| Suite | Result |
| --- | --- |
| M0 | 26 passed, 1 optional Alem skipped; 5.44 seconds |
| M1 | 41 passed, 2 optional Alem skipped; 8.71 seconds |
| M2 | **195 passed, 2 optional Alem skipped; 21.64 seconds** |
| M2 Ruff formatting/lint and strict mypy | Passed; 41 source/test files |

The commands are the same checks used by Core CI:

```sh
uv sync --locked --python 3.12
uv run --no-sync python -m ruff format --check mosaic tests
uv run --no-sync python -m ruff check mosaic tests
uv run --no-sync python -m mypy mosaic tests
uv run --no-sync python -m pytest -q -m "not alem"
```

On this Windows host, source mypy and fresh workspace temporary directories were used.
PowerShell was launched outside the package so its temporary script did not become an
untracked source file during provenance hashing. Initial attempts with such temporary files
failed the provenance read; the clean full reruns above passed. No source workaround was
applied to M0/M1.

Coverage retains compiler/private-state boundaries, all-pass promotion, persistence and
retrieval integrity, failure accounting, teacher-free validation/reuse and routing/gate
invariants. New tests check independent default seeds, five usage dimensions, missing values,
phase aggregation, absence of teacher usage in new dimensions and CI configuration.

Optional pinned-Alem integration previously passed its two simulator/routing smokes.
Those results concern adapter plumbing, not task intelligence or skill acquisition. Real
provider normalization and prompt-protocol checks now live in [M1.25](../../milestone-1.25/README.md).

## Independent default seed demonstration

`configs/skill-pilot.yaml` now uses acquisition **42**, validation **19** and reuse **20**.
`configs/skill-reuse.yaml` also uses seed 20. All three seeded fake resets produce
`part_position=7`; selected initiation fields match. This is a **seen-condition test**, not
near transfer or held-out generalization. Seed independence does not create a novel condition.

The updated default pilot was executed into a fresh local directory:

```sh
uv run --locked python -m mosaic.experiments.learn --config configs/skill-pilot.yaml --output-dir runs/skill-pilot-independent-seeds
```

Result: `succeeded: validated_skill_reused`; `skill-38cb7756ffe8c336` was promoted.

| Phase | Weak calls | Teacher calls | Environment actions |
| --- | ---: | ---: | ---: |
| Acquisition, seed 42 | 2 | 16 | 18 |
| Validation, seed 19 | 16 | 0 | 16 |
| Pipeline reuse, seed 20 | 16 | 0 | 16 |
| Pipeline total | **34** | **16** | **50** |

One deterministic compilation incurred zero compiler-model calls. The 16-action procedure
still incurs a weak inference call per validation/reuse decision. Tokens are synthetic fixture
units; they are not an API price estimate or scientific cost-benefit result.

The fresh output contains `events.jsonl`, `summary.json`, `playbook.json`, candidate/report
artifacts and full episode evidence in `acquisition/events.jsonl`,
`validation/19/events.jsonl` and `reuse/20/events.jsonl`. Existing output directories are
refused. To use the standalone reuse example after this alternative output location, point
its PlayBook path at that run; the normal fresh-clone `runs/skill-pilot` example remains valid.

## Usage and research interpretation

`TokenUsage` now preserves optional input, output, reasoning, cached-input and provider-total
counts, provider request ID and measurement source. Missing dimensions remain `None`.
Budgets and phase merges retain known subtotals and unknown-call counts; they do not double
count reasoning/cache or infer a provider total. Legacy summaries supply no evidence of zero
for newly added dimensions. Fake providers remain deterministic.

The unchanged M2 Alem smoke adapter and minimal prompt do not establish real-model capability.
The M1.25 calibration adapter supplies reviewed official rules and documented solo settings;
it is separate from M2 skill semantics.

The corrected sequence is M0 complete, M1 complete, M2 engineering scaffold complete,
M1.25 measurement/provider hardening, M1.5 real-model capability calibration, M1.75 protocol
freeze, M2-R real-trajectory skill semantics, M3 hard teacher withdrawal and M4 economics.
M2 does not justify proceeding directly to M3. No full-task or dynamic-subgoal redesign,
withdrawal study, learned compiler or investment gate is implemented here.

## Next recommended step

**Run real-model calibration.** A useful gap based on real paired executions must precede
protocol freeze, the M2-R representation decision and any hard teacher-withdrawal study.
