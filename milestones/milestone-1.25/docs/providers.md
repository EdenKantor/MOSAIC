# Provider and measurement semantics

The Ollama, Groq and Gemini Developer API adapters are implemented. Their tests use mocked
network transports. Model identifiers do not establish a useful competence gap, transfer or
economic savings. The current API-only candidate protocol and free-only execution boundaries
are defined in [the calibration contract](free-only-calibration.md); the earlier Ollama/Groq
configs are preserved preparation, not an empirically selected pair.

## Requests and secrets

All adapters serialize the same actor-visible request: system instruction, public goal,
step, current observation, advertised action names/descriptions and bounded visible feedback.
Private environment snapshots and verifier evidence do not enter the request. Calibration
uses the official solo Alem rules and the same configured eight-turn history for both roles.
Prompt serialization omits framework agent IDs and role labels from current/history context;
those identities remain in audit records rather than instructions about how an actor should
behave.
See [the prompt comparison](alem-prompt-comparison.md) for declared differences from upstream.

Groq uses the fixed HTTPS chat-completions endpoint and obtains its bearer credential only
from `GROQ_API_KEY` in the process environment. The key is not a configuration, trace or model
request field. Ollama uses `OLLAMA_HOST`, defaulting to `http://127.0.0.1:11434`, and permits
only a loopback HTTP(S) origin without credentials, query, fragment or endpoint path. It sends
no Groq credential to Ollama. Neither adapter bundles or downloads model weights.

Gemini uses a fixed HTTPS `generateContent` endpoint and obtains `GEMINI_API_KEY` only from
the process environment, in the `x-goog-api-key` header rather than a URL. It receives the
same system/user content and requests native JSON MIME type without an action schema enum.
No search, paid tools, explicit cache creation, Vertex API or billing API is used.

Provider errors have static messages that omit response bodies and credential-bearing
exceptions. Normalized strings and identifiers redact both captured/current provider
credentials. This is a defined artifact boundary, not permission to place secrets in prompts
or a guarantee against arbitrary external code reading the process environment.

## One attempted call, one POST

`StdlibHttpTransport` opens one fresh connection, sends one non-streaming POST and closes it.
There is no SDK retry layer, automatic reattempt, fallback model or redirect following. HTTP
redirects, rate limits and other non-success status codes are explicit failures. Error bodies
are not normalized into actor output or exception messages.

Proactive free-only batch guards run before call reservation. An allowed episode call then
reserves its role-labelled call before dispatch and records elapsed latency.
A failed request remains one attempted call. Its unavailable usage stays unknown; it does not
become a zero-token success. A returned provider response still must satisfy metadata/usage
validation and the ordinary exact JSON action parser. Parsing failure consumes the decision
and is not repaired by another provider request. The runner does not claim cancellation of
work already dispatched when a local timeout occurs.

Core tests inject `HttpTransport` fakes for payloads, responses, failures and malicious values.
They make no real HTTP requests and require no credentials. Optional service preflight and
real-model calibration are distinct from these unit tests. Preflight makes no inference call.
It may inspect local Ollama tags and the installed Alem pin. Groq preflight checks credential
presence only and makes no remote network request; it does not validate account/model access.

## Raw usage dimensions

Every normalized `TokenUsage` preserves these independent dimensions:

| Dimension | Ollama response field when present | Groq response field when present |
| --- | --- | --- |
| Input | `prompt_eval_count` | `usage.prompt_tokens` |
| Output | `eval_count` | `usage.completion_tokens` |
| Reasoning | `reasoning_tokens` | `usage.completion_tokens_details.reasoning_tokens` |
| Cached input | `prompt_eval_cached_count` | `usage.prompt_tokens_details.cached_tokens` |
| Provider total | `total_tokens` | `usage.total_tokens` |

The adapter does not infer optional dimensions from response text, reasoning text, input/output
arithmetic or another provider's conventions. An absent field stays `None`. Zero is retained
only when the provider explicitly reports zero. Counts must be nonnegative integers; booleans,
negative counts and inconsistent cached-input counts are rejected. No raw hidden reasoning
content is inserted into actor history.

The official [Ollama chat API](https://docs.ollama.com/api/chat) normally exposes aggregate
evaluation counts, rather than a separate reasoning-token count. That dimension therefore
stays unknown unless explicitly supplied. The Groq mappings follow its
[chat-completions API](https://console.groq.com/docs/api-reference),
[reasoning documentation](https://console.groq.com/docs/reasoning) and
[prompt-cache usage fields](https://console.groq.com/docs/prompt-caching).

`provider_request_id` records an available response/body/header identifier. Measurement labels
distinguish `provider`, `synthetic` and `unavailable`; `measurement_source` names the parsed
response source even when no dimensions are available. Fake fixture counts remain synthetic
and cannot be converted to API costs.

Aggregate and weak/teacher role totals preserve each dimension, its known subtotal and its
unknown-call count. If any constituent call has a missing dimension, that dimension's complete
total is `None`; known subtotals remain useful lower bounds. Legacy summaries missing a new
dimension supply no evidence of zero and are treated as unknown for their recorded calls.
Provider-reported totals are preserved. Reasoning and cached input may be subsets of other
dimensions: do not add them again. No cross-provider tokenizer equivalence or monetary/GPU
cost is synthesized. Decision caps do not enforce lifetime token spending.

Recording failures attempt to recover known call reservations and usage from the episode
trace. If full ledger coverage cannot be established, `budget_complete=false` marks the
report incomplete and complete token totals remain unknown. Recorded call counts and known
subtotals then describe only the recovered portion; they are not proof that unrecorded
inference cost was zero.

## Declared model controls

The historical calibration configuration in
[milestone-1.5/configs/calibration-pilot.yaml](../../milestone-1.5/configs/calibration-pilot.yaml)
uses temperature 0 and a 2,048-token output cap for both roles. Ollama sends `think=true` and
`options.num_ctx=32768`; Groq sends `reasoning_effort="medium"`. These are deliberately
provider-specific controls, not identical mechanisms or a matched deliberation budget.
The requested context window also does not imply equal model context capacities.

The providers send JSON output controls, and MOSAIC separately enforces one advertised action
through its parser. A provider may still fail to emit usable JSON or consume its output budget
before an action. Preserve those failures during calibration rather than changing prompts,
output caps or retry rules after inspecting a score.

Independent candidate and paired calibration contains no skills, PlayBook, learning or
withdrawal intervention. Empirical candidate selection must precede registration of a
pair-specific five-seed pilot. M1.75 and M2-R require useful real calibration evidence; M3
remains stopped. See the current contract for Gemini usage fields and cumulative quota guards.
