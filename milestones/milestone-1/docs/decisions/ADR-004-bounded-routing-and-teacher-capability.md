# ADR-004: Finite weak windows and an explicit teacher capability

Status: accepted for Milestone 1.

Use a replaceable synchronous escalation-policy protocol driven by immutable counters and
environment outcomes. The first policy ends a weak window after consecutive objective failures
or a finite goal-success deadline, permits bounded weak recovery, then switches the remaining
episode to a separately attributable teacher.

Gate all runtime teacher calls through a capability whose withdrawn state holds no provider.
Preserve the provider and environment protocols. Build model requests from visible values only;
do not serialize snapshots or verifier internals into prompts. Reset ephemeral history per run.

Keep aggregate budgets for existing consumers and add per-role totals. Add role fields to generic
call/response events rather than duplicating teacher invocation logs. Version the milestone's
event schema as 2; Milestone 0 retains its schema-1 reader in its preserved folder.

The finite deadline avoids inventing a progress detector but can route a still-progressing weak
agent to the teacher. Document and expose limits in experiment configuration. No additional
dependencies or persistent knowledge components are introduced.
