# ADR-001: Typed JSONL experiment records

Status: accepted for Milestone 0.

Research results require auditable raw trajectories, not only success totals. Persist every
observable decision and transition as a typed frozen event in sequence. Store explicit failure
and dispatch events so accounting remains possible when providers or environments fail.
JSONL requires no service and remains readable by analysis tools and a future renderer.

Event payloads are discriminated by kind and validated against the envelope type. Flush each
write and refuse existing run directories. This is append-only application storage, not a
cryptographically tamper-proof log or a simulator restored exclusively from events.
