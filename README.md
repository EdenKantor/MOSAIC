# MOSAIC

Research on whether temporary teacher access can produce durable agent capability under a
fixed lifetime inference budget. The current decision caps do not yet enforce that lifetime
token or monetary budget.

Each milestone lives in a separate, independently runnable folder and receives its own Git
commit. Earlier milestone folders are preserved when later milestones are implemented.

| Folder | Scope | Status |
| --- | --- | --- |
| [milestones/milestone-0](milestones/milestone-0/README.md) | Deterministic single-agent vertical slice | Implemented and tested |
| [milestones/milestone-1](milestones/milestone-1/README.md) | Bounded teacher escalation, routing-only baseline B3 | Implemented and tested |
| [milestones/milestone-2](milestones/milestone-2/README.md) | Candidate procedures, weak validation and shared PlayBook | Engineering scaffold complete; scientifically unvalidated |
| [milestones/milestone-1.25](milestones/milestone-1.25/README.md) | Usage accounting, real-provider adapters and independent calibration runner | Implemented; real-provider executions pending |
| [milestones/milestone-1.5](milestones/milestone-1.5/README.md) | Paired real-model capability protocol and configurations | Prepared; calibration results pending |

Milestone 2 demonstrates plumbing, validation gates, persistence and accounting with
deterministic fixtures. It does not demonstrate useful procedural transfer between real models. Its
default acquisition, validation and reuse seeds are 42, 19 and 20 respectively; all select the
same initial part location (7). Different seeds under matching starting conditions remain
seen-condition reuse rather than held-out generalization.

Run the engineering demonstration from its directory:

```sh
cd milestones/milestone-2
uv sync --locked
uv run --locked python -m mosaic.experiments.learn --config configs/skill-pilot.yaml --output-dir runs/skill-pilot
```

Each implemented runtime folder has its own package, lockfile, configurations, tests and
documentation. M1.5 is currently a protocol/configuration folder using the M1.25 runtime.
Do not combine the milestone packages into one Python environment.

The roadmap now requires calibration before a withdrawal study:

| Order | Milestone | Scope |
| --- | --- | --- |
| 1 | M0 — Complete | Deterministic single-agent vertical slice |
| 2 | M1 — Complete | Bounded teacher routing |
| 3 | M2 — Engineering scaffold complete | Procedure gates, persistence and accounting; scientific claims pending |
| 4 | M1.25 — Measurement / Provider Hardening | Real-provider interfaces and trustworthy inference accounting |
| 5 | M1.5 — Real-Model Calibration | Establish useful weak/teacher behavior before skill claims |
| 6 | M1.75 — Protocol Freeze | Register tasks, splits, prompts, budgets, baselines and evaluation rules |
| 7 | M2-R — Real-Trajectory Skill Semantics | Resolve activation/validation semantics using calibrated trajectories |
| 8 | M3 — Hard Teacher Withdrawal | Measure retention after a defensible registered boundary |
| 9 | M4 — Economics | Cost-aware investment and full acquisition/reuse economics |

The current teacher-suffix compiler describes the state at teacher takeover, while retrieval
and validation start from reset. The scripted waiting prefix masks this mismatch. Real weak
progress can make the candidate inapplicable or invalid from reset. This is a blocker for
real-trajectory skill claims; see [the semantics and redesign options](milestones/milestone-2/docs/skills.md).
The existing `TeacherCapability.withdraw()` is a tested runtime primitive, not a completed M3
experiment.

Core CI covers M0, M1, M1.25 and M2 on Python 3.12 without Alem or real-provider calls.
Optional Alem integration and real-model calibration remain separate from deterministic core
checks. The next recommended step is to **run real-model calibration** after provider and
measurement hardening, before freezing the protocol or implementing M2-R/M3.
