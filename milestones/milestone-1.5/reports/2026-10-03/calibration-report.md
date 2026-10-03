# Free-only real-model calibration — 2026-10-03

**Partial real execution; both provider batches stopped. No empirical Weak/Teacher pair selected.**

The user verified Gemini's project as `No billing account` and Groq's account as `Free — $0 — Current Plan`. Both environment credentials were available; values were never printed, logged or persisted. No billing/account settings were changed, paid feature used, or paid fallback attempted.

## Access and non-inference preflight

Five metadata GETs and the local pinned-Alem/configuration checks preceded inference. The original [access preflight](access-preflight.json) records zero inference; [expanded registration](expanded-access-preflight.json) preserves the subsequent registry change after initial smoke. Metadata listing is weaker evidence than successful generation. The safety/moderation classifiers requested for exclusion were never candidates.

Groq's published free limits for GPT-OSS 20B/120B and Qwen 3.8 27B were 30 RPM, 1,000 RPD, 8,000 TPM and 200,000 TPD. Returned request headers describe RPD and token headers describe TPM; they do not establish unused daily tokens. Actual account/model pools can differ. Gemini project/model RPM/RPD/TPM/TPD and remaining quotas were unknown and remain null. [Groq limits](https://console.groq.com/docs/rate-limits), [Gemini limits](https://ai.google.dev/gemini-api/docs/rate-limits).

Eligible Gemini standard text IDs were registered after reviewing [official free pricing](https://ai.google.dev/gemini-api/docs/pricing); two hosted Gemma 4 IDs also followed [the official API interface](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api). Native JSON compatibility for the unexecuted Gemma candidates remains unverified. Allam's active listing does not establish free eligibility: the current published free-limit table omitted it, its price was pending and its 4K context needs feasibility review. It was skipped before inference; bilingual specialization alone does not make it incapable.

## Candidate smoke and availability

Smoke used seed 42 and one advertised action, not a goal-completion test. All seven usable responses produced an accepted Move West action. One-step truncation is expected and excluded from capability scoring. Requested and returned served-model IDs are separate in the JSON evidence.

| Provider | Exact requested model | Smoke/access result | Total POST attempts |
| --- | --- | --- | ---: |
| gemini | `gemini-2.5-flash-lite` | HTTP_404 | 1 |
| gemini | `gemini-3.5-flash-lite` | SMOKE_PASSED | 1 |
| gemini | `gemini-3.1-flash-lite` | SMOKE_PASSED | 1 |
| gemini | `gemini-2.5-flash` | HTTP_404 | 1 |
| gemini | `gemini-2.5-pro` | HTTP_404 | 1 |
| gemini | `gemini-3.5-flash` | SMOKE_PASSED | 1 |
| gemini | `gemini-3.6-flash` | SMOKE_PASSED | 1 |
| gemini | `gemini-3.7-flash` | HTTP_503 | 1 |
| gemini | `gemini-3.8-flash` | NOT_DISPATCHED_PROVIDER_STOPPED | 0 |
| groq | `openai/gpt-oss-20b` | SMOKE_PASSED | 3 |
| groq | `openai/gpt-oss-120b` | SMOKE_PASSED | 3 |
| groq | `qwen/qwen3.8-27b` | SMOKE_PASSED | 2 |
| groq | `allam-2-7b` | SKIPPED_ACCESS_UNVERIFIED | 0 |
| gemini | `gemini-3-flash-preview` | NOT_DISPATCHED_PROVIDER_STOPPED | 0 |
| gemini | `gemma-4-26b-a4b-it` | NOT_DISPATCHED_PROVIDER_STOPPED | 0 |
| gemini | `gemma-4-31b-it` | NOT_DISPATCHED_PROVIDER_STOPPED | 0 |

Gemini's three model-specific HTTP 404s were preserved and not retried. A correction allowed first attempts on different registered models rather than treating one unavailable ID as proof all Gemini models were unavailable. HTTP 503 then stopped Gemini entirely. Later registered IDs were never dispatched, so their generation access and capability remain unknown.

## Empirical task diagnostic

Every diagnostic independently reset pinned Alem, seed 42, with the same visible rules, observations, legal actions, eight-turn history, temperature 0, 2,048 output cap and 200-step task cap. Both GPT-OSS candidates used reasoning effort medium. Initial snapshot, action-list and visible-prompt hashes match for their resource-acquisition pair, and recorded reset traces match preparation. Other provider defaults and tokenizers are not equivalent deliberation budgets.

| Task family / seed | GPT-OSS 20B success | GPT-OSS 120B success | Qwen 3.8 27B | Gemini candidates | Family label |
| --- | --- | --- | --- | --- | --- |
| Resource acquisition / 42 | 1/1 | 1/1 | Unknown: HTTP 429 | Not diagnosed | CEILING: single-seed signal |
| Navigation + acquisition / 42 | Not executed | Not executed | Not executed | Not diagnosed | AMBIGUOUS |
| Simple crafting / 42 | Not executed | Not executed | Not executed | Not diagnosed | AMBIGUOUS |
| Multi-step crafting / 42 | Not executed | Not executed | Not executed | Not diagnosed | AMBIGUOUS |

| Completed model/task | Actions | Calls | Input | Output | Reasoning | Cached input | Provider total | Episode wall s | Inference wall s |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `openai/gpt-oss-20b` / collect_wood / 42 | 2 | 2 | 4403 | 170 | 138 | Unknown | 4573 | 62.281 | 1.159 |
| `openai/gpt-oss-120b` / collect_wood / 42 | 2 | 2 | 4403 | 333 | 304 | Unknown | 4736 | 114.563 | 1.545 |

Both complete trajectories were Move West, then Do, with success checked against the official COLLECT_WOOD achievement. The observed success difference on this seed is **0 percentage points**. One seed cannot establish population success rates or reject the same-family pair on harder tasks. The 120B trajectory used more reported output/reasoning tokens here; this is not evidence of a stronger Teacher. Monetary request cost was not reported and stays unknown.

The Qwen diagnostic made one physical request and received HTTP 429 with no accepted action and unknown usage. Its original normalized category was `billing`. The legacy classifier could mistake a generic billing URL or upgrade advice for paid-access evidence. Because raw error bodies were intentionally discarded, the exact historical cause cannot be proved or relabeled retrospectively. Historical trace/category and provider stop remain intact. Future classification now prioritizes quota/rate codes unless the response explicitly establishes a paid-access requirement. No paid upgrade or retry followed.

Infrastructure-incomplete episodes and zero-dispatch rows are excluded from capability denominators; their costs/unknown usage are retained. Failure decision indices are not accepted environment actions. There were no observed invalid JSON, rejected actions or timeouts among these attempts. The one-step smoke cap is not diagnostic step-cap saturation.

## Accounting and retry policy

There were **16 inference POST attempts** (8 Gemini, 8 Groq), **11 returned model responses**, **11 accepted environment actions**, **7 usable model smokes** and **2 complete capability episodes**. Five failed attempts have unknown token usage. No hidden SDK retry, parser-repair request, redirect, paid fallback or account mutation occurred. One generate invocation permits exactly one remote inference attempt; any future retry must be explicitly recorded as another physical attempt.

TokenUsage fields are input_tokens, output_tokens, reasoning_tokens, cached_input_tokens, provider_total_tokens, provider_request_id, measurement and measurement_source. Missing provider fields remain null. Gemini normalizes usageMetadata; Groq normalizes chat.completions usage; deterministic fake usage remains explicitly synthetic. Reasoning/cache dimensions are not added again to provider totals. Ollama is implemented and mocked but was not used in this API-only execution.

| Provider | Known input subtotal | Unknown input attempts | Known output subtotal | Unknown output attempts | Known provider-total subtotal | Unknown total attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gemini | 7948 | 4 | 30 | 4 | 8503 | 4 |
| groq | 14844 | 1 | 857 | 1 | 15701 | 1 |

These are known subtotals, not complete charged-usage totals. Full reasoning/cache unknown counts, request IDs, returned served IDs and per-call latency are in [calibration-summary.json](calibration-summary.json). Provider pacing was at least 61 seconds between request starts, with hydrated preceding ledgers, conservative physical caps and Groq token planning guards. The 8,000-token next-request reserve is a planning assumption, not a measured tokenizer upper bound. Account-wide remaining quota was unavailable; HTTP 429 stopped dispatch even though local use was small.

## Prompt comparison and scientific blockers

The [pinned source comparison](../../../milestone-1.25/docs/alem-prompt-comparison.md) follows Alem's official solo instruction wrapper, level-gated rules and visible solo affordances. Actor input contains no private snapshots, RNG state or verifier evidence. Intentional differences include focused achievement goals, one player, soft specialization with both efficiencies 1.0, strict JSON, history serialization, provider reasoning controls and no retries. This is a grounded calibration setup, not a claim of official cooperative-benchmark equivalence.

The [registrations](../../configs/) include original and expanded one-step smoke and four-family diagnostic configurations. The three additional Gemini IDs and the increase from 25 to 75 physical diagnostic requests were registered after initial smoke; task caps, visible protocol and sampling were unchanged. The expanded theoretical scope is 11,200 requests, distinct from the cumulative 75-call/provider allowance and never authorized as one unattended batch. The original Ollama/Groq role-labeled pilot remains historical preparation, not an empirically selected pair.

**No pair is recommended from these results.** Weak headroom and materially better Teacher behavior have not been demonstrated. The five-seed pilot was not run; M1.75 is not frozen; M2-R was not implemented. M2 is preserved as deterministic procedure plumbing, not scientific transfer evidence. Its Teacher-suffix initiation/reset-only retrieval mismatch remains a blocker. Full-task workflow skills are the preferred later design for Experiment #1, pending real calibration; no initiation predicate was weakened. M3 remains stopped.

Remaining blockers are verified usable free quota/provider availability, completion of candidate diagnostics on harder task families, and a defensible empirically selected pair before registering the five-seed pilot. Allam requires independent free-eligibility and context feasibility evidence before dispatch; unexecuted Gemma JSON compatibility remains unknown. No automatic recovery or paid upgrade is part of the protocol.

## Engineering checks and Laboratory

Python 3.12 locked synchronization, Ruff formatting/lint and mypy passed for all four core folders. Offline tests: M0 26; M1 41; M2 195; M1.25 265; Laboratory 16 — **543 offline passes**. A separate pinned-Alem protocol check passed once with zero inference, bringing verified test passes to 544. Ordinary CI installs no Alem/JAX and makes no provider calls. Hosted CI status is reported separately after publication.

The local read-only [Laboratory](../../../laboratory/README.md) provides event timeline/replay speed and matched-task/seed side-by-side comparison. Public projection excludes private simulator state; privileged audit is a separately acknowledged view. Runtime role labels are accounting annotations, not empirical roles. It has no inference or billing controls. HTTP/API and JavaScript replay checks passed; browser visual interaction was unavailable. No saved public image frames exist for these traces. Raw normalized runs remain local under runs/real-model-20261003; this committed projection preserves accounting/outcomes and source integrity hashes.

## Evidence and exact files

- [calibration-summary.json](calibration-summary.json): complete public per-attempt accounting, exact candidate inventory, task results and raw source hashes.
- [selection-decision.json](selection-decision.json): explicit null pair, family labels and closed downstream gates.
- [changed-files.json](changed-files.json): exact repository-relative paths in this continuation, grouped by publication scope.

## One recommended next step

Resume real-model calibration when verified free quota and provider availability permit.
