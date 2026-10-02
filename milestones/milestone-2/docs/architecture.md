# Current architecture

Milestone 2 composes persistent procedures with the existing environment-independent runner.
This is a completed engineering scaffold for plumbing, validation gates, persistence and
accounting, using deterministic fixtures. Scientific usefulness and transfer are unvalidated.

```text
frozen configuration
       |
experiments/learn.py: acquisition → compilation → validation → promotion → reuse
       |
experiments/runner.py
       |-- core/agent.py: visible requests, skill hint and action parsing
       |-- core/routing.py: bounded weak/recovery policy
       |-- core/teacher.py: revocable runtime teacher capability
       |-- skills/: models, compiler, validator, PlayBook and retrieval
       |-- providers/: async protocol and stateless fake fixtures
       |-- environments/: adapter protocol, fake pump task and pinned Alem adapter
       |-- core/events.py: immutable typed JSONL, schema 3
       |-- core/budget.py: aggregate and per-role usage
       `-- core/provenance.py: source and dependency fingerprints
```

The ordinary episode CLI remains available. Configurations without skills follow the original
weak-only or routing-only path. The pipeline CLI composes stages; the runner knows no fake-pump
mechanics or Alem APIs. The environment remains authoritative for validity and goal success.

Domain values use frozen Pydantic models, tuples and scalar fields. Snapshots use canonical JSON
text; provider SDK values stay outside the research core. Configuration is validated once,
including procedure bounds and predetermined validation seeds.

## Information and capability boundaries

Weak and teacher providers are distinct instances. The teacher remains behind
`TeacherCapability`, which drops its provider reference on withdrawal and checks new calls
before dispatch. History, routing counters, skill cursor and ledger are local to an episode.
No previous history is silently carried into a new agent run.

`TeacherCapability.withdraw()` is a tested primitive that prevents new invocations through
that capability. It is not a scheduled hard-withdrawal experiment, a sandbox against arbitrary
Python, or cancellation of an already dispatched request. M3 is deferred until measurement,
real-model calibration, protocol freeze and real-trajectory semantics are complete.

Models receive task, visible observation, advertised actions and bounded visible feedback.
Weak reuse additionally receives the selected procedure and cursor, without validation verdicts
or source provenance. Snapshots and private verifier evidence remain audit data. The compiler
consumes sanitized teacher execution plus the verified-success gate. It cannot inspect snapshots
to invent activation conditions.

Validation constructs fresh weak-only episodes with no teacher capability or escalation policy.
Retrieval is deterministic matching, not inference. Compilation also uses no model provider.

## Persistence and execution

Candidate compilation extracts accepted teacher actions into a bounded declarative sequence.
Scope and exact visible initiation conditions constrain use. Validation records whether the
weak agent followed the complete sequence and achieved verified success. Promotion requires
every configured validation seed to pass.

The current sequence is a teacher suffix whose initiation conditions describe takeover.
Retrieval happens only during episode initialization, while validation also starts at reset.
The fake weak prefix waits, so selected state fields still match despite changed excluded step
metadata. Real weak progress can break this equivalence. The engine does not retrieve mid-episode
or reconstruct takeover states for validation. This is a blocker for real-trajectory skill
semantics, not a demonstrated transfer mechanism.

Two later options are documented in [skill semantics](skills.md): a complete accepted
reset-to-goal workflow under initial conditions (preferred for Experiment 1), or dynamically
retrieved subgoal suffixes validated from defensible starting states. Neither redesign is
implemented here; real-model calibration comes first.

The JSON PlayBook stores only verified entries with reports and integrity metadata. Atomic
replacement prevents a partial file from becoming valid. This is a single-writer prototype,
not a concurrent database. Reopening validates schema, candidate digest and report consistency;
it does not authenticate provenance or prove the report's truth. Retrieval checks
environment/revision/task and visible initiation conditions.

Execution means constructing a request, paying for weak inference, parsing the proposal and
dispatching through the ordinary interface. The store never replays actions for free.
Retrieval misses continue ordinary execution. A divergent or rejected action removes the
skill hint and records a failure; ordinary execution may continue next turn. Validation instead
fails immediately on that condition. Only environment verification establishes success.

## Traces and accounting

Schema 3 preserves role-labelled model/action events and adds typed candidate, validation,
promotion, retrieval and execution failure events. Acquisition, each validation episode and
each reuse episode retain full traces. The pipeline trace describes stage outcomes; it does not
replace episode evidence.

Each call begins with unknown usage. Responses provide usage when available; failed calls retain
unknown tokens and measured latency. Aggregate calls and known subtotals reconcile weak and
teacher groups. Environment dispatch attempts count even when rejected.

Compilation/retrieval operation counts and local time, together with exact UTF-8 injected skill
bytes, keep memory overhead visible. Pipeline reporting separates acquisition, validation and
reuse rather than omitting acquisition cost from reuse comparisons.

There is no population scheduler, communication layer, learned routing, vector store, repair
service, cost-aware investment gate, global withdrawal phase or renderer. Milestones 0 and 1
retain independent source, event schemas and lockfiles.

The revised roadmap is M0 complete → M1 complete → M2 engineering scaffold complete → M1.25
measurement/provider hardening → M1.5 real-model calibration → M1.75 protocol freeze → M2-R
real-trajectory skill semantics → M3 hard withdrawal → M4 economics. The next recommended step
is to run real-model calibration after provider and measurement hardening.
