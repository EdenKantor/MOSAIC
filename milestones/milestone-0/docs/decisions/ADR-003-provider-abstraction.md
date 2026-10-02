# ADR-003: Async provider calls and explicit unknown usage

Status: accepted for Milestone 0.

Use an async generate protocol returning provider-independent request/response models. Reserve
every call before invocation, then attach usage when it becomes available. Missing token counts
remain unknown, with known subtotals preserved. Never estimate absent usage as zero.

Only a deterministic fake provider exists. Its fixture counts are explicitly synthetic.
External providers, timeouts, retry policies and monetary budget enforcement are deferred.
