# Experiment semantics

## Inputs and outputs

YAML is validated once into deeply frozen configuration. Unknown fields, invalid seeds, unknown
tasks and uninspected environment revisions are rejected. `max_steps` bounds decision turns,
including malformed proposals; it is not a token or monetary budget enforcement mechanism.
Lifetime inference-budget enforcement belongs to a later milestone with real model usage.

Each fresh run directory contains:

- `events.jsonl`: raw typed events, full configuration, exact requests/actions/responses,
  environment verification and snapshots, terminal events and usage totals.
- `summary.json`: outcome, reason, decision count, final verification and budget.

`ExperimentStarted` records the MOSAIC commit, dirty state, source checksum, lockfile checksum,
installed dependencies, Python/platform and concrete environment/provider implementations.
The manifest also identifies the teacher capability, underlying teacher provider and routing
policy. Configuration records environment revision, sampling, model identities, initial teacher
access and routing limits. Every request contains its role, prompt, task, visible observation,
current action definitions, output limit and bounded episode feedback. Schema version is 2.

With no `teacher` or `routing` section, the original weak-only runner path remains available.
Routing requires both sections. `routing.yaml` demonstrates teacher takeover;
`weak-success.yaml` and `recovery-success.yaml` finish without teacher calls;
`teacher-withdrawn.yaml` records blocked escalation. Their policies are scripted fixtures.

The manifest fingerprint identifies uncommitted source but does not archive it. Preserve that
checkout or commit it alongside experimental traces. Installed dependencies are also recorded;
the lockfile is necessary to rebuild a matching environment. Stochastic/cloud reproducibility
is not guaranteed by a seed alone, and no cloud provider is implemented here.

## Determinism and replay

`tests/test_experiments.py::test_deterministic_trace` compares all trace content except UUIDs,
timestamps, measured model latency and elapsed wall time. It does not discard substantive
configuration, prompts, actions, usage, provenance or snapshots.
`tests/test_routing.py::test_routing_is_deterministic` applies the same comparison to teacher
routing. A reused-runner test separately checks that a new episode receives no previous history.

Replay is verified by resetting the fake environment with the recorded seed/task, applying
`ActionDispatched` events in sequence and comparing every recorded snapshot. No replay CLI or
snapshot-restoration interface is implemented. Alem snapshots include its complete environment
state and advanced PRNG key; replay of Alem itself is not yet tested across machines.

## Budget interpretation

`model_calls` counts attempted calls. `environment_actions` counts dispatch attempts.
`weak` and `teacher` contain separate call/token/latency totals, unknown counters and known
subtotals. Aggregate calls and token subtotals sum those groups. `teacher_call_fraction` divides
teacher calls by all calls, or is null when no model was called. Escalations and blocked
escalations are explicit summary fields; a blocked invocation consumes no teacher call.
Token totals are null if any constituent call has unknown usage for that dimension. Known
subtotals and unknown-call counts remain available. A failed provider call records unknown
usage, not zero. Fake usage is explicitly marked `synthetic`, using whitespace counts of the
serialized request and response; it is unsuitable for API cost estimation or model comparison.

`wall_clock_ms` measures execution inside the runner through outcome recording, before final
summary serialization. It includes provenance collection, reset, calls, simulation and event
writes; it excludes package installation and adapter/provider construction. Model-response
and role-specific failed-call events record elapsed provider time. Per-role wall time is the
sum of provider-call latencies, not an allocation of simulator or trace-writing time. GPU time,
estimated API cost, compilation, retrieval and communication costs are not claimed or synthesized
in this milestone.

## Failure semantics

Provider errors and simulator exceptions produce `RunErrored`, failure and terminal events when
the recorder is writable. Unknown actions, malformed JSON and extra model-output fields produce
`ActionRejected` and consume a bounded turn. Only `EnvironmentAdapter.verify()` can establish
success. A model claim, an accepted action, a reward or a terminal flag cannot establish success.

CLI exit codes are 0 success, 1 unsuccessful completed episode, and 2 setup/runtime error.
An existing output directory is never overwritten.
