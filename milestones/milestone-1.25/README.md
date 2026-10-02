# MOSAIC — Milestone 1.25

**Measurement and Provider Hardening.**

This independent folder adds local Ollama and hosted Groq provider adapters, five-dimensional
usage accounting, and a grounded Alem prompt for real-model calibration. The implementation
is **uncalibrated**: mocked provider checks and deterministic fixtures do not establish model
competence, useful transfer or savings. M0–M2 remain independently runnable.

## Install and check without inference

Use Python 3.12 for the pinned Alem extra. From this folder:

```sh
uv sync --locked --extra alem --python 3.12
uv run --locked --extra alem python -m mosaic.experiments.run --calibration-config ../milestone-1.5/configs/calibration-pilot.yaml --preflight
```

Preflight checks configuration and runtime prerequisites without calling a model. It does not
prove that either model can solve a task. It can inspect local Ollama model tags; for Groq it
checks credential presence without contacting the remote endpoint or validating access.
Groq credentials come only from `GROQ_API_KEY` in the
process environment. Ollama uses `OLLAMA_HOST` or its loopback default; the adapter accepts only
a loopback HTTP(S) origin. Do not put credentials in YAML, prompts or tracked files.

The calibration configuration names local `qwen3:4b` and Groq `openai/gpt-oss-120b`. These are
configured model identifiers, not claims that either service is available or that their weak /
teacher relationship is established. No model weights are bundled.

## Run the pilot

```sh
uv run --locked --extra alem python -m mosaic.experiments.run --calibration-config ../milestone-1.5/configs/calibration-pilot.yaml --output outputs/calibration-pilot
```

The pilot evaluates weak-only and teacher-only arms independently. It has no escalation,
skills, PlayBook, persistent actor memory or withdrawal intervention. Four task families
(`collect_wood`, `collect_stone`, `make_wood_pickaxe`, `make_stone_pickaxe`) use seeds
42, 19, 20, 7 and 11. Each arm receives the same public task, visible observations, legal
solo actions and configured eight-turn visible history. Each episode permits at most 200
decisions. The pilot schedules 40 episodes; this is a bounded calibration, not a lifetime
token or monetary budget.

Both arms use temperature 0 and a 2,048-token output cap. Ollama explicitly uses thinking
on and context window 32,768; Groq uses reasoning effort `medium`. Those controls have different
provider semantics. This is an intentional declared difference, not an assertion of identical
deliberation or tokenizer budgets. Compare raw usage and failures, and preserve the exact
configuration with the traces.

Review the five-seed pilot before using the 20-seed expansion:

```sh
uv run --locked --extra alem python -m mosaic.experiments.run --calibration-config ../milestone-1.5/configs/calibration-expanded.yaml --confirm-pilot-reviewed --output outputs/calibration-expanded
```

The confirmation flag records the required review gate; it does not certify model competence.
Output directories must be fresh. Calibration outcomes, invalid proposals, transport failures
and unknown usage remain visible; there are no hidden retries to improve scores.

The output contains `config.json`, `summary.json`, `report.md` and per-arm evidence under
`<family>/<seed>/<weak|teacher>/`. Initial snapshot, advertised-action and visible-prompt hashes
must match across a pair and its recorded reset evidence before it enters comparable outcome
aggregates. Mismatches/errors remain recorded and their returned costs remain in raw totals.
CLI exit 0 means the calibration protocol completed, including valid goal failures; exit 2
marks setup/runtime or comparability errors.

## Measurement and protocol

Each provider invocation makes one non-streaming standard-library HTTP POST. There is no SDK,
automatic retry or redirect following. Provider failure consumes an attempted call with unknown
usage when no valid response usage is available. Strict JSON action parsing uses the advertised
list; rejected proposals consume a decision rather than triggering a repair call.

Accounting retains input, output, reasoning, cached input and provider-reported total token
dimensions separately. Missing dimensions remain `None`, with known subtotals and unknown-call
counts in each role and in aggregate. Reasoning and cache counts are not added again to total
usage; provider totals are preserved rather than synthesized. Request IDs and measurement
sources support audit. Local elapsed provider time is not API cost or GPU time.

The new Alem calibration adapter uses official solo rules and level-based disclosure, precise
visible coordinates and legal affordances. Its solo action filter removes Request/Give from
both actor-visible lists. Soft specialization with both efficiencies 1.0 avoids the legacy
solo-warrior hard gate on pickaxe tasks. These conditions are the same for both roles and
differ from the cooperative benchmark. See [the pinned prompt comparison](docs/alem-prompt-comparison.md).

Read [provider and usage semantics](docs/providers.md), [the verification report](docs/validation.md)
and [the calibration protocol folder](../milestone-1.5/README.md).

## Offline core checks

```sh
uv sync --locked
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked python -m mypy mosaic tests
uv run --locked python -m pytest -q
```

Ordinary core tests use mocked transports and deterministic environments. They require no
credentials, model calls or Alem installation. Real Alem adapter checks are optional and
separate. Exact executed commands and results belong in the verification report.

The revised order is M0 complete → M1 complete → M2 engineering scaffold complete → M1.25
measurement/provider hardening → M1.5 real-model calibration → M1.75 protocol freeze → M2-R
real-trajectory skill semantics → M3 hard withdrawal → M4 economics. M2's engineering scaffold
remains scientifically unvalidated. The next recommended step is to **run real-model calibration**.
