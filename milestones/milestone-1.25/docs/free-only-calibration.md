# Free-only candidate calibration contract

## Evidence and billing

The user manually verified Gemini's project has **No billing account** and Groq's
account has **Free — $0 — Current Plan** on 2026-10-03. This permits free inference
within available quotas. It never permits upgrades, billing mutations, prepayment,
paid-only models/tools or fallback. Credentials are read only into the child
process environment; credential values, hashes and raw HTTP bodies are never artifacts.

Metadata GETs list candidates without inference. Standard text eligibility is
checked against [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) and
[Groq limits](https://console.groq.com/docs/rate-limits). Listed/active status is not
proof of remaining quota or successful inference. Gemini's current numeric project
limits require its [active quota dashboard](https://ai.google.dev/gemini-api/docs/rate-limits);
unknown values remain null. The current Groq table lists GPT-OSS 20B, 120B and Qwen
3.8 27B with 30 RPM, 1,000 RPD, 8,000 TPM and 200,000 TPD; actual account limits may differ.

Allam is active with 4,096 context, but omitted from the current published free-limit
table. Its [official bilingual description](https://console.groq.com/docs/model/allam-2-7b)
does not establish inability in English; context feasibility and free eligibility
must be verified before including it. Safety classifiers are explicitly excluded.

## Public interface and controls

Same task/seed reset, rules, observations, actions, prompt protocol, temperature,
output cap and history policy apply to all candidates. Gemini maps the same
system/user content into systemInstruction/contents. It uses responseMimeType JSON
without a schema enum; Groq uses json_object. Local parsing measures malformed JSON,
wrong shape and illegal action separately. No repair prompt or second remote call
occurs for one generate. Unsupported controls fail explicitly, never fall back.

GPT-OSS sizes both use medium reasoning effort. Other registered models retain
provider defaults, recorded in configs; no equivalence of deliberation is asserted.
Gemini thinking-budget/level controls are available for later preregistered protocols,
not selected after inspecting a score. Returned model version is retained separately
from the requested ID. Future runs must recheck mutable service deployments.

## Physical attempts and quota boundaries

The stdlib transport uses a new connection for one POST and follows no redirects.
No SDK or automatic retry exists. HTTP error content is bounded, reduced to safe
categories and discarded. Quota, rate limit, access, timeout, server, billing,
payment, insufficient-credit and free-unavailable failures stop the provider batch.
Errors and rejected proposals remain visible; they do not justify another prompt.
A nonfatal model-specific 404 persists that model as unavailable and is never retried;
it does not imply that independent registered models on the provider are unavailable.
Explicit fatal 404 and unknown-cost/invalid-response stops remain conservative.

Before each scope, record episodes × max_steps plus the separate physical cap.
Unknown quotas permit at most one episode per provider/invocation. Cumulative prior
summaries carry attempted calls, known usage, pacing/quota information and stop
state forward. Token dimensions unavailable after a request cannot be assumed zero.
Proactive request/usage/rate guards act before accounting reservation and POST;
cap exhaustion is infrastructure incomplete. Never treat it as a capability failure.
Numeric Groq request headers refer to RPD; token headers refer to TPM. Reset durations
are normalized seconds; raw headers are not persisted. Waiting before a new allowed
request is pacing, not retrying a failed request. Failed batches never auto-resume.

## Usage schema

| Normalized field | Gemini field | Groq field |
| --- | --- | --- |
| input_tokens | usageMetadata.promptTokenCount | usage.prompt_tokens |
| output_tokens | usageMetadata.candidatesTokenCount | usage.completion_tokens |
| reasoning_tokens | usageMetadata.thoughtsTokenCount | usage.completion_tokens_details.reasoning_tokens |
| cached_input_tokens | usageMetadata.cachedContentTokenCount | usage.prompt_tokens_details.cached_tokens |
| provider_total_tokens | usageMetadata.totalTokenCount | usage.total_tokens |
| provider_request_id | responseId/header when available | response/header ID when available |
| measurement_source | gemini.generateContent.usageMetadata | groq.chat.completions.usage |

Absent values remain None. Explicit zero is known. Provider totals remain raw;
neither cache nor reasoning is added again. Cross-provider tokenizer units are not
assumed equal. Provider cost stays unknown when not returned; free-account status
supports the separate paid-usage assertion, not a fabricated response cost.

## Research gates

ENGINEERING_VERIFIED means offline protocol/accounting behavior passed checks.
EMPIRICALLY_OBSERVED applies only to recorded real interactions, not fixtures or metadata.
HYPOTHESIS includes proposed role hierarchies. NOT_TESTED includes useful transfer,
withdrawal retention and investment savings. A useful real capability gap and
registered paired pilot must precede M1.75/M2-R. M3 is stopped regardless of provider access.
