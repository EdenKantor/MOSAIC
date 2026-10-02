# MOSAIC — Milestone 1.5

**Real-Model Calibration Protocol — Uncalibrated.**

This folder contains the pilot and expansion configurations for independent weak-only and
teacher-only calibration. The runtime lives in the separately runnable
[Milestone 1.25 folder](../milestone-1.25/README.md). The configurations are ready for
calibration; no completed real-model study or useful competence gap is claimed.

Calibration contains no skills, PlayBook, learning, teacher escalation or withdrawal
intervention. Its purpose is to establish useful weak and teacher behavior under the same
grounded task interface before protocol freeze and real-trajectory skill redesign.

## Protocol

| Condition | Declared setting |
| --- | --- |
| Task families | `collect_wood`, `collect_stone`, `make_wood_pickaxe`, `make_stone_pickaxe` |
| Pilot seeds | 42, 19, 20, 7, 11; five per family and role |
| Arms | Weak-only and teacher-only, each reset independently |
| Weak identifier | Ollama `qwen3:4b` |
| Teacher identifier | Groq `openai/gpt-oss-120b` |
| Common sampling | Temperature 0; output cap 2,048 tokens |
| Decision cap | 200 per episode |
| Weak deliberation | Thinking on; context window 32,768 |
| Teacher deliberation | Reasoning effort `medium` |
| Actor context | Official solo Alem rules, legal actions, visible observations and eight-turn history |
| Expansion | 20 seeds per family and role, only after pilot review |

These provider controls are intentional recorded differences. Local thinking and hosted
reasoning effort are not identical mechanisms or a matched deliberation/tokenizer budget.
Model identifiers are configuration values rather than proof of availability or competence.
No model weights or credentials are included.

The Alem environment is pinned to `14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e`. One-player
calibration has no coordination or god mode. Soft specialization with both efficiencies 1.0
unblocks the focused resource/tool tasks for the solo warrior. Both arms receive the same
public information. The official prompt uses precise coordinates, legal affordances and
current-level rule disclosure. Private snapshots and goal-verifier evidence remain audit data.
See [the pinned prompt comparison](../milestone-1.25/docs/alem-prompt-comparison.md).

## Run from the runtime folder

Use Python 3.12 for the optional Alem dependency. Run these commands from `milestone-1.25`:

```sh
uv sync --locked --extra alem --python 3.12
uv run --locked --extra alem python -m mosaic.experiments.run --calibration-config ../milestone-1.5/configs/calibration-pilot.yaml --preflight
uv run --locked --extra alem python -m mosaic.experiments.run --calibration-config ../milestone-1.5/configs/calibration-pilot.yaml --output outputs/calibration-pilot
```

Preflight verifies prerequisites without inference. Groq requires `GROQ_API_KEY` in the process
environment. Ollama uses its default loopback host or `OLLAMA_HOST`; credentials never belong
in YAML or tracked files. Output directories must be fresh. A preflight pass does not mean a
model can solve the tasks. Local model tags may be inspected; the Groq check validates
credential presence without contacting the remote endpoint or proving access.

Review task success, invalid actions, failure cases, step-cap saturation and raw usage by role
and family before expanding. Preserve negative outcomes and record any calibration-driven
family selection explicitly before protocol freeze. Then run:

```sh
uv run --locked --extra alem python -m mosaic.experiments.run --calibration-config ../milestone-1.5/configs/calibration-expanded.yaml --confirm-pilot-reviewed --output outputs/calibration-expanded
```

The review flag is required for expansion; it does not certify scientific readiness. There
are no automatic provider/parse retries, fallback models, reflection calls or hidden teacher
repairs. Every attempted call and dispatched action remains accounted for. Missing usage
dimensions stay unknown; reasoning/cache counts are not added again to provider totals.
See [usage semantics](../milestone-1.25/docs/providers.md) and
[the verification/calibration status](../milestone-1.25/docs/validation.md).

Each run preserves the configuration, summary and readable report plus per-family/seed/role
episode evidence. Snapshot, advertised-action and visible-prompt hashes must match for both
arms and their recorded reset evidence. Incomparable pairs are retained with explicit errors
and excluded from outcome aggregates; their returned costs stay visible. A completed run can
contain legitimate goal failures. It is not a certification that all tasks succeeded.

## Gate for later work

The pilot is looking for a useful empirical region, not a predetermined hierarchy. Weak
success around 20-50% and teacher success around 60-90% are planning examples, not scientific
thresholds. The weak model must have substantial room to improve while remaining capable;
the teacher must be materially better; neither should be at ceiling. A pair scoring 3%/6%
would be rejected for a shared floor, and a pair scoring 94%/97% for a shared ceiling.
Treat the family labels as hypotheses about difficulty until real trajectories support them.

No real executions have filled this required evidence table:

| Task family | Weak success | Teacher success | Weak cost/usage | Teacher cost/usage |
| --- | --- | --- | --- | --- |
| Resource acquisition | Pending | Pending | Unknown | Unknown |
| Navigation + acquisition | Pending | Pending | Unknown | Unknown |
| Simple crafting | Pending | Pending | Unknown | Unknown |
| Multi-step crafting | Pending | Pending | Unknown | Unknown |

Do not resume skill-learning work until real executions establish a useful gap. Then freeze
the experimental protocol and decide the M2-R procedural representation from real trajectories.
Full-task workflows are the preferred starting design for Experiment #1; teacher-suffix/subgoal
skills require dynamic activation and a defensible activation-state validation design. Neither
redesign is implemented here.

M0 and M1 are complete, and M2's engineering scaffold is complete but scientifically
unvalidated. The revised sequence is M1.25 measurement/provider hardening → M1.5 real-model
calibration → M1.75 protocol freeze → M2-R real-trajectory skill semantics → M3 hard teacher
withdrawal → M4 economics. The teacher-suffix/reset activation mismatch must be resolved in
M2-R; the existing withdrawal primitive is not a completed M3 study.

The next recommended step is to **run real-model calibration** and review the five-seed pilot.
