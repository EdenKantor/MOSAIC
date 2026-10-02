# Experiment semantics

Milestone 2 is a completed engineering scaffold, not a scientifically validated skill study.
Deterministic fixtures exercise plumbing, validation gates, persistence and accounting;
successful runs do not establish useful transfer or real-model learning.

## Inputs and episode outputs

YAML is validated once into frozen configuration. Unknown fields, invalid seeds, unsupported
tasks and uninspected environment revisions are rejected. `max_steps` bounds decision turns,
including malformed proposals; it does not enforce lifetime token or monetary budgets.

The ordinary episode command remains:

```sh
uv run --locked python -m mosaic.experiments.run --config configs/routing.yaml
```

Each fresh output directory contains `events.jsonl` and `summary.json`. Events record exact
requests, responses, actions, snapshots, verification, configuration, terminal outcomes and
usage. `ExperimentStarted` records commit/dirty state, source and lockfile checksums, installed
dependencies, Python/platform and concrete implementation identities. Schema is 3.

Configurations without skills retain Milestone 1 behavior. Teacher routing still requires
paired teacher/routing input; weak-only configurations need neither. Persisted reuse mode in
`configs/skill-reuse.yaml` reads `runs/skill-pilot/playbook.json` and supplies no teacher.

## Acquisition, validation and reuse

```sh
uv run --locked python -m mosaic.experiments.learn --config configs/skill-pilot.yaml --output-dir runs/skill-pilot
```

The pilot runs the existing bounded routing acquisition. Verified goal success and accepted
teacher actions are required before compilation. The compiler consumes visible requests and
action feedback; private snapshots and verifier evidence are excluded.

The candidate extracts the bounded teacher sequence and exact conditions for configured visible
JSON fields. The pilot uses `position`, `part_position`, `has_part` and `pump_repaired`,
a 32-action cap, acquisition seed `42`, validation seed `19` and reuse seed `20`. All three
select initial part location 7 and matching selected starting conditions. Validation seeds are nonempty
and unique. Weak-only validation and reuse each allow up to 32 decisions; reuse uses
`agent_1`.

Every validation seed creates a fresh weak-only episode. Promotion requires complete candidate
execution and verified success on all seeds. Missing initiation, divergence or a rejected action
fails validation immediately; fallback success cannot qualify. The shared PlayBook stores
the promoted entry and report, then is reopened for new reuse.

Outputs under the selected fresh directory are:

| Path | Meaning |
| --- | --- |
| `events.jsonl`, `summary.json` | Pipeline stages, outcome and accounting |
| `acquisition/` | Teacher-assisted episode trace and summary |
| `candidate.json` | Scoped candidate and source provenance |
| `validation/<seed>/` | Raw weak-only validation episode |
| `validation.json` | All-seed promotion evidence |
| `playbook.json` | Promoted entry and integrity metadata |
| `reuse/<seed>/` | Raw weak-only reuse episode |

The pipeline trace complements the per-episode traces. It cannot substitute for reconstructing
all inference and action attempts.

## Matching, determinism and persistence

A retrieved entry must be verified, intact and match environment identity, pinned revision,
task and exact configured visible starting conditions. No embeddings or teacher inference
participate. Default reuse repeats seen conditions under an independent seed; different part
locations do not match. The seed split is not held-out generalization.

The candidate is a teacher suffix with initiation measured at takeover. The runner attempts
retrieval only at reset, and validation starts at reset too. In this fixture, the weak prefix
waits and changes only excluded step metadata, so projected takeover and reset conditions
coincide. A real weak prefix can move or manipulate the task before escalation, making a valid
teacher suffix inapplicable or unsuccessful from reset. Current experiments do not solve this
semantic blocker. [Skill semantics](skills.md) records the preferred complete-workflow option
for Experiment 1 and the dynamic-subgoal alternative; neither is implemented.

A miss continues ordinary weak/routing behavior. Once retrieved, proposal divergence, parsing
rejection or environment rejection records a skill failure and removes the hint. Ordinary
execution may continue next turn. Completing the procedure ends hint injection. The environment
verifier determines success independently.

Deterministic comparisons exclude timestamps, UUIDs and measured elapsed times. Configuration,
conditions, procedure content, prompts, actions, usage and snapshots remain substantive.
Source provenance changes with the checkout; preserve the committed code and lockfile alongside
traces. A seed alone does not guarantee stochastic/cloud reproducibility.

Existing fake replay checks reset with the recorded seed/task, dispatch recorded actions and
compare snapshots. No replay CLI or snapshot-restoration API is introduced. Alem snapshots
retain full state and PRNG key; cross-machine Alem replay is not claimed.

## Budget interpretation

An attempted model call counts once even if it fails. A blocked teacher invocation adds no
teacher call. Role totals retain calls, known token subtotals, unknown counts, synthetic usage
and provider time. Aggregate model totals reconcile those groups. Environment actions count
actual dispatch attempts.

Token totals are null when any constituent call has unknown usage in that dimension; known
subtotals remain available. Fake usage uses synthetic whitespace counts of the serialized full
request/response. This includes skill prompts but is not a tokenizer or API cost.

Reports separate acquisition, validation and reuse inference. Compilation and retrieval record
operation counts and local elapsed time without inventing model calls or tokens. Exact UTF-8
bytes injected as skill content measure prompt overhead; do not add them a second time to
provider input usage.

Runner wall time includes provenance collection, reset, inference, simulation and trace writes
through outcome recording, before summary serialization. Role wall time sums provider latencies,
not simulator time. Compilation/retrieval time stays distinct from provider usage. GPU time
and API cost are not synthesized.

Zero teacher calls during reuse does not establish savings. Acquisition, compilation,
validation, retrieval and prompt overhead belong in any later amortized comparison.

## Failure semantics

Only `EnvironmentAdapter.verify()` establishes success. Model claims, rewards, accepted actions
and terminal flags are insufficient. Invalid proposals consume a decision turn and produce
rejection events without hidden retries. Provider/simulator errors produce error and terminal
events when recording remains possible; filesystem errors propagate.

Candidate rejection, validation, promotion, retrieval and procedure execution failure are typed
auditable events. Persisted integrity failures produce explicit errors and retain retrieval
attempt overhead. They do not silently become retrieval misses.
Existing output directories are never overwritten.

CLI exit codes remain 0 success, 1 completed unsuccessful experiment and 2 setup/runtime error.

## Calibration before later milestones

Core CI covers M0, M1 and M2 on Python 3.12 without Alem or ordinary real-provider calls.
These checks validate engineering behavior, not the research hypothesis. Optional integration
and real-model calibration evidence must be reported separately without fabricated results.

The revised order is M0 complete → M1 complete → M2 engineering scaffold complete → M1.25
measurement/provider hardening → M1.5 real-model calibration → M1.75 protocol freeze → M2-R
real-trajectory skill semantics → M3 hard withdrawal → M4 economics. The tested capability
withdrawal primitive is preserved; no M3 experiment is implemented in this folder.

The next recommended step is to run real-model calibration after provider and measurement
hardening.
