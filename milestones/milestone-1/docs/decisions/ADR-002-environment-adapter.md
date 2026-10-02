# ADR-002: Synchronous environment contract

Status: accepted for Milestone 0.

Expose reset, observe, available_actions, step, verify and snapshot through a small protocol.
Compose the runner with an implementation. Environment state and deterministic verification
are authoritative; model text never determines success. Keep simulation synchronous.

Fake mechanics and all Alem translation are isolated in their adapters. Concrete action labels
are sufficient for current tasks. Parameter schemas, teams and snapshot restoration will be
introduced only when an actual experiment requires them.
