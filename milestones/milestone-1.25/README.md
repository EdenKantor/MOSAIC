# MOSAIC — Milestone 1.25

**Measurement, provider hardening and independent calibration infrastructure.**

Fake, optional loopback Ollama, Groq and Gemini Developer API adapters share the
same public actor request. Exact models live in configurations. No model hierarchy,
skill-transfer result or withdrawal result follows from this engineering scaffold.
M0, M1 and M2 remain separately runnable and preserved.

## Offline verification

```sh
uv sync --locked --python 3.12
uv run --no-sync python -m ruff format --check mosaic tests
uv run --no-sync python -m ruff check mosaic tests
uv run --no-sync python -m mypy mosaic tests
uv run --no-sync python -m pytest -q -m "not alem"
```

Core tests use deterministic environments and mocked HTTP; they make zero network
calls and require no credentials. Alem/JAX is an optional extra, outside ordinary
PR installation and tests. Core CI also checks the read-only Laboratory separately.

## Calibration entry point

Follow the [API candidate registration and scientific gates](../milestone-1.5/README.md).
The historical Ollama/Groq pilot configs remain preparation evidence. Real paired
calibration now requires free-tier attestation, explicit physical allowances and
a prior empirical selection decision/evidence digest. Do not run those old configs
as an assumed Weak/Teacher comparison.

Credentials come only from `GEMINI_API_KEY` and `GROQ_API_KEY` in process memory.
No billing or account-setting API is used. The operator must establish free-only
account status before inference; model metadata alone does not establish it.
Preflight checks local prerequisites with zero inference. Separately recorded
metadata GETs establish listed access, while real smoke checks actual inference.

Each logical generate makes at most one physical POST. No SDK retries, redirects,
re-prompts or fallback models are used. Rejected JSON/actions consume a decision.
Fatal provider failures stop the relevant batch, preserving partial traces and unknown
usage. Proactive batch guards run before call reservation and provider dispatch.
They record an incomplete infrastructure prefix rather than a failed task outcome.
A nonfatal model-specific 404 excludes that model without retry while permitting
independent registered models on the same provider to receive their first attempt.

## Protocol and accounting

Independent arms receive identical grounded goals, reset conditions, public rules,
visible observations, advertised actions and eight-turn feedback history. Public
prompt/action and private reset audit hashes check parity. Actors receive neither
snapshots, verifier evidence, full achievements, PRNG state nor framework role labels.
No skills, PlayBook, escalation, learning or teacher withdrawal are used.

Gemini and Groq use native JSON mode without advertised-action schema constraints;
the common local parser tests exact one-action shape and legality. This preserves
the ability to measure illegal proposals. Provider reasoning settings remain
declared differences rather than assumed equivalent budgets.

Normalized usage preserves input, output, reasoning, cached input and provider-total
tokens, request ID and measurement source. Missing fields remain `None`; known
subtotals and unknown-call counts survive aggregation. Reasoning/cache dimensions
are never added again to provider totals. Returned served-model IDs and sanitized
numeric quota metadata support audit; no provider price is invented.

The Alem pin is `14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e`. Official solo rule
disclosure, precise visible coordinates and solo legal affordances are used.
Soft specialization with both efficiencies 1.0 prevents a solo class gate from
artificially blocking pickaxe goals. See [the official prompt comparison](docs/alem-prompt-comparison.md),
[provider semantics](docs/providers.md) and [validation](docs/validation.md).

M2 demonstrates plumbing, validation gates, persistence and accounting with
deterministic fixtures. It does not demonstrate useful procedural transfer between
real models. Its teacher-suffix/reset retrieval mismatch remains a blocker;
calibration must precede M1.75, M2-R and any M3 study.
