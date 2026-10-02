# Validated shared procedures

Milestone 2 makes the learning boundary explicit:

```text
verified teacher-assisted episode
              |
deterministic candidate compilation
              |
fresh weak-only validation on configured seeds
              |
all episodes follow the complete procedure and pass the environment verifier
              |
atomic promotion to the shared PlayBook
              |
scoped retrieval and weak-model execution in a new episode
```

## Candidate representation and compilation

A candidate is a finite declarative sequence of advertised action names. It carries
initiation conditions, allowed tools, the goal termination condition, environment/revision/task
scope, source model/agent/episode provenance and a content hash. It is not executable Python.
The configured action cap prevents an unbounded generated program.

The compiler accepts only a currently verified successful acquisition containing accepted
teacher actions. It receives sanitized visible requests and action feedback, not snapshots
or private verifier evidence. It uses the verified-success verdict as a gate; it cannot infer
success from a model claim. Weak waiting turns are excluded from the teacher procedure.

Compilation is deterministic extraction. There is no compiler model, reflection call,
generalization, parameter inference or hidden teacher planner. The candidate preserves a
demonstrated action sequence under explicit starting conditions. Compilation operations and
elapsed local time are recorded.

Initiation is exact equality on configured top-level fields from visible JSON observations.
The pilot uses `position`, `part_position`, `has_part` and `pump_repaired`. These describe
the visible teacher starting point. Excluding a field from the predicate does not prove that
field irrelevant to the procedure.

## Validation and promotion

Validation seeds are predetermined, nonempty and unique. Each seed creates a fresh
environment, weak provider and runner. Validation has no teacher capability and no escalation
policy. It cannot silently repair a candidate through teacher access.

The fixed promotion rule requires every validation episode to pass both checks:

1. The weak agent follows the entire candidate procedure.
2. The environment verifier confirms the goal.

A missing initiation match, divergent proposal, parsing failure or rejected action ends
validation as a failure. A generic fallback success cannot promote the candidate. Rejections
and all validation outcomes remain auditable. The configured seeds provide evidence only
about the tested conditions.

The PlayBook persists verified entries together with their validation reports. Candidate
data alone cannot become retrievable. Atomic JSON replacement prevents partial writes.
The store is intended for one writer; concurrent writers are outside this milestone's
contract. Reopening checks schema, candidate content digest and validation consistency.
It cannot establish that a stored validation report is true; reports must come from the trusted
experiment runner. Candidate hashes detect procedure changes, not forged provenance or reports.
They are not signatures or a security boundary against code with filesystem access.

## Retrieval and execution

Retrieval is local and deterministic. A skill must match environment identity, pinned revision,
task and exact visible initiation conditions. Retrieval does not consult a teacher, use
embeddings, summarize a trace or search private simulator state. Only verified entries qualify.
A retrieval miss continues the ordinary weak/routing path.

A selected skill supplies its ID/version, procedure content and cursor to the weak request.
Validation verdicts and source provenance are not actor hints. The provider is called once
per decision; the advertised-action parser and environment dispatch remain authoritative.
The store never bypasses these interfaces by replaying actions automatically.

Divergence from the hinted action, parsing rejection or environment rejection records a
skill execution failure and removes its hint. Ordinary weak/routing execution may continue
on the next turn. Completing the sequence also ends injection. Neither completing a skill
nor receiving an accepted action establishes success: the goal verifier remains authoritative.

The fake `follow_skill` policy chooses the supplied action and otherwise waits. This proves
prompt delivery and execution accounting, not that a language model can interpret a procedure.
The pilot reuses another agent ID with the same seed and visible layout. A near-condition
mismatch refuses retrieval.

## Scope and interpretation

The original runtime teacher gate remains in force for acquisition. Compilation, validation
and retrieval receive no teacher provider. Weak-only reuse establishes absence of a teacher
in that episode. There is no predefined global withdrawal phase or population retention study.

The procedure has no loops, branches, variables, parameter binding, autonomous repair,
composition or learned investment rule. Exact conditions favor refusal at the cost of coverage.
Malformed, altered or out-of-scope entries must fail integrity or matching checks rather than
become unrestricted instructions.

All-pass promotion is the first fixed-rule shared-procedure scaffold for baseline B5. A
successful scripted seen-condition demo does not establish durable capability capital,
economic benefit, near transfer or superiority over a matched-budget weak baseline.
