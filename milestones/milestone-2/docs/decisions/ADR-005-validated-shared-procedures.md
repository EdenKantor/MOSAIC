# ADR-005: Validated shared declarative procedures

Status: Accepted for Milestone 2.

## Context

A successful teacher intervention is not sufficient evidence that a weak agent can reuse it.
Arbitrary generated code would introduce an execution surface beyond the current named-action
interface. Reuse must expose inference, validation and memory overhead.

## Decision

Extract accepted teacher actions from a verified successful acquisition into a bounded
declarative procedure. Compile sanitized visible observations and execution feedback. Use
the success verdict as a gate without reading private simulator evidence.

Scope each candidate to environment, pinned revision and task. Initiation is exact equality
on configured visible JSON fields. Preserve source provenance, allowed tools, goal termination
and a content hash.

Validate on predetermined nonempty unique seeds in fresh weak-only runners, with no teacher
capability or routing policy. Promote only when every episode follows the entire procedure and
the environment verifies success. Persist verified entries with reports in an atomic,
single-writer JSON PlayBook and check integrity on reopen.

Retrieve by exact scope/condition matching. Inject the selected procedure and cursor into
ordinary weak requests. Each action retains a model call, parsing, advertised-action validation
and environment dispatch. Reuse failures remove the hint; validation failures reject promotion.

Record compilation/retrieval counts and local time, exact injected skill bytes, and separate
acquisition, validation and reuse inference budgets.

## Consequences

This is a transparent fixed-rule scaffold for B5. It avoids generated Python, embeddings and
compiler-model dependencies. Visible compiler inputs and teacher-free validation/retrieval
make hidden teacher assistance harder to introduce.

Exact conditions reduce coverage. Finite sequences do not express parameterized transfer,
loops or composition. All-pass validation supports only tested seeds. The first demo repeats
seen conditions with scripted providers, giving wiring evidence rather than real-model learning
or economic benefit.

Atomic replacement does not solve concurrent writers. Candidate hashes and schema checks do
not authenticate provenance or prove stored validation reports true; reports come from trusted
runner execution. Global withdrawal, skill repair and cost-aware investment remain later milestones.
