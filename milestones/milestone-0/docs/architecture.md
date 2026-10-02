# Current architecture

The implementation is deliberately small:

```text
configs/*.yaml -> core/config.py (frozen validated input)
                       |
experiments/run.py -> experiments/runner.py
                       |-- core/agent.py: request construction and action parsing
                       |-- providers/base.py: async generate protocol
                       |     `-- providers/fake.py: deterministic fixture
                       |-- environments/base.py: synchronous environment protocol
                       |     |-- environments/fake.py: seeded pump task
                       |     `-- environments/alem/: optional pinned Alem integration
                       |-- core/events.py: typed append-only JSONL
                       |-- core/budget.py: per-call usage and raw totals
                       `-- core/provenance.py: source/dependency fingerprints
```

The runner receives an environment and provider through composition. It imports neither
fake-environment mechanics nor Alem APIs. The command-line entry point is the composition
root, selecting the requested implementations. No provider SDK types enter core models.

Domain values use frozen Pydantic models, tuples and scalar fields. Adapter-specific snapshots
use canonical JSON text to avoid mutable dictionaries inside frozen events. Actions currently
have a single concrete name; parameterized tools are intentionally deferred.

The runner observes, obtains current action specifications, records an exact model request,
calls the provider, records usage, validates the proposed action, dispatches it, records state,
and asks the environment verifier about success. Termination and success are distinct.
Bad outputs consume a model call and a decision step. They do not trigger hidden retries or
fallback actions. The environment may reject an action that passed the advertised-action check.

Each call reservation begins with unknown usage; a response replaces it with reported usage.
A failed call therefore remains counted with unknown tokens. Environment actions count actual
dispatch attempts, including calls that are rejected or raise. Ledger totals are in the final
events and summary. Per-call usage retains provider/model identity and measurement provenance.

The event recorder writes exclusively to a new file and flushes each event. Events have UTC
timestamps, UUIDs, schema version, contiguous sequence numbers, experiment/episode/agent IDs,
step, event type and discriminated typed payloads. Recording errors are propagated; the system
does not pretend an experiment is auditable if the filesystem fails.

There is no database, web server, rendering layer, teacher, skill system or communication layer.
