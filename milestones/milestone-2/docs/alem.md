# Alem integration

Inspected official source: [alem-world/alem-env](https://github.com/alem-world/alem-env),
commit `14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e`, package version 0.2.1, MIT license.

The inspection covered `README.md`, `AGENTS.md`, `LICENSE`, `pyproject.toml`,
`examples/llm_text_smoke.py`, `examples/random_rl_agent.py`, `alem/alem_env.py`,
`alem/llm/alem_language_wrapper.py`, symbolic environment implementations, action masking,
state definitions and achievement enums. The dependency extra pins that exact commit; the
adapter checks installed VCS provenance rather than trusting a floating package version.

The official API creates an environment through `make_alem_env`, resets its language wrapper
with a JAX PRNG key, and steps a synchronous list of player actions. Observations contain long-
and short-term text contexts. Current valid actions come from `compute_action_mask`; success
is read directly from the `COLLECT_WOOD` achievement bit in environment state.

This integration uses the official cooperative symbolic implementation with `player_count=1`,
coordination disabled, specialization disabled, no communication, no rendering and no model
calls inside Alem. The wrapper performs only deterministic observation formatting. Its
language action names map to the actual upstream action enum. The adapter rejects unadvertised
actions before dispatch. Mask acceptance is conservative and does not prove an action changed
the world. Noop can advance time without earning an achievement.

All Alem imports and translation stay in `mosaic/environments/alem/`. The same `ExperimentRunner`
executes fake and Alem tasks. The original optional smoke requests `collect_wood` and takes a
single scripted Noop. The Milestone 1 routing smoke takes two weak Noops (attempt and recovery),
then one teacher Noop. Both complete with expected goal failure. Milestone 1's validation recorded
two passing optional tests against the pinned real package on Windows CPU; the routing test checks
role calls, action count, escalation and verifier failure. Those historical checks are wiring tests,
not comparable to Alem's official three-agent benchmark. Current executed checks belong in the
Milestone 2 validation report. No adapter mechanics were changed for routing.

The optional dependency is restricted to Python 3.12 by upstream's `<3.13` requirement; CPU
Windows wheels for pinned JAX/JAXlib 0.4.38 were installed in this session. No Alem source is
vendored. Using an external package preserves upstream's bundled MIT license.

Milestone 2 leaves this adapter unchanged. Its named-field initiation compiler requires visible
JSON fields; Alem currently exposes text contexts. No meaningful Alem procedure acquisition or
transfer is demonstrated by the fake-pump skill pilot. Supporting structured, grounded Alem
activation conditions requires an explicit later design rather than inventing private-state hints.

No real model provider is added in Milestone 2. A successful language-model-controlled Alem task
remains future work. The earlier Ollama availability check was negative; it is not a current
service-health guarantee.
