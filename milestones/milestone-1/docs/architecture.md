# Current architecture

Milestone 1 preserves the single-agent vertical slice and adds bounded routing through composition:

```text
configs/*.yaml -> core/config.py (frozen validated input)
                       |
experiments/run.py -> experiments/runner.py
                       |-- core/agent.py: visible requests and action parsing
                       |-- core/routing.py: bounded weak/recovery policy
                       |-- core/teacher.py: runtime inference capability gate
                       |-- providers/base.py: async generate protocol
                       |     `-- providers/fake.py: stateless deterministic fixtures
                       |-- environments/base.py: synchronous environment protocol
                       |     |-- environments/fake.py: seeded pump task
                       |     `-- environments/alem/: optional pinned Alem integration
                       |-- core/events.py: typed append-only JSONL, schema 2
                       |-- core/budget.py: aggregate and per-role raw accounting
                       `-- core/provenance.py: source/dependency fingerprints
```

The runner receives an environment, weak provider, optional teacher capability and optional
escalation policy. It imports neither fake-environment mechanics nor Alem APIs. The CLI composes
distinct provider instances and wraps the teacher in `TeacherCapability`. Core models contain
no provider SDK types. Routing limits and initial teacher access are immutable experiment inputs.

Domain values use frozen Pydantic models, tuples and scalar fields. Adapter-specific snapshots
use canonical JSON text to avoid mutable dictionaries inside frozen events. Actions currently
have a single concrete name; parameterized tools are intentionally deferred.

The model receives visible observation, task, advertised actions and bounded visible episode
feedback; snapshots and verifier evidence remain audit data. No prior trace is read.

The runner observes, obtains current action specifications, records an exact model request,
calls the provider, records usage, validates the proposed action, dispatches it, records state,
and asks the environment verifier about success. Termination and success are distinct.
Bad outputs consume a model call and a decision step. They do not trigger hidden retries or
fallback actions. The environment may reject an action that passed the advertised-action check.

After a finite weak window fails, the policy permits bounded weak recovery. Failure after all
recovery windows makes escalation eligible. An available teacher then controls the remaining
episode through the same action/parser/environment interfaces and shared decision cap. A withdrawn
teacher stops the run with an explicit blocked event and zero new teacher calls. Both the runner
and capability check access. See [routing.md](routing.md) for exact signals and gate limits.

History, routing counters and ledger are local to each run. Fake providers are stateless.

Each role-specific call reservation begins with unknown usage; a response replaces it with reported usage.
A failed call therefore remains counted with unknown tokens. Environment actions count actual
dispatch attempts, including calls that are rejected or raise. Ledger totals are in the final
events and summary, separately for weak and teacher roles. Per-call usage retains provider/model
identity, latency and measurement provenance. Failed calls also record measured latency.

The event recorder writes exclusively to a new file and flushes each event. Events have UTC
timestamps, UUIDs, schema version, contiguous sequence numbers, experiment/episode/agent IDs,
step, event type and discriminated typed payloads. Recording errors are propagated; the system
does not pretend an experiment is auditable if the filesystem fails.

There is no persistent learning, database, web server, rendering layer, skill store, shared memory,
population or communication layer. Milestone 0 remains separately runnable with its original
schema 1 reader and independent lockfile.
