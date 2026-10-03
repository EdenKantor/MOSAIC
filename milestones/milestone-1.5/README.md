# MOSAIC — Milestone 1.5

**Real-model capability calibration. No empirical Weak/Teacher pair has been selected.**

The API-only candidate protocol uses the [M1.25 runtime](../milestone-1.25/README.md).
It contains no skills, PlayBook, learning, escalation or withdrawal. The earlier
Ollama/Groq pilot and expansion files are retained as historical preparation;
their role labels are hypotheses, and those files must not bypass candidate selection.

The current registrations are [expanded provider smoke](configs/free-model-smoke-expanded-20261003.yaml)
and [expanded candidate diagnostics](configs/free-model-selection-expanded-20261003.yaml).
They retain nine stable Gemini text IDs, add legacy Gemini 3 Flash and two hosted
Gemma 4 text models, and register Groq GPT-OSS 20B, GPT-OSS 120B and Qwen 3.8 27B.
Allam remains an explicit free-access skip. Gemini 2.5 Flash-Lite's initial inference
returned HTTP 404 despite its metadata listing; it is not retried. Model sizes and
names do not assign roles. The original registrations remain unchanged evidence.
The [expanded access inventory](reports/2026-10-03/expanded-access-preflight.json) records exact IDs,
metadata status, public limits, unknown remaining quota and exclusions.

The [partial real results](reports/2026-10-03/calibration-report.md) record seven usable
one-action smokes and two successful resource-acquisition diagnostics, one each for
GPT-OSS 20B and 120B on seed 42. Both providers subsequently stopped: Gemini HTTP 503,
Groq HTTP 429. The original Groq error category was `billing`; a legacy classifier could
mistake a generic billing link for paid-access evidence. Its raw body was discarded,
so the exact cause cannot be established retrospectively. The classifier is fixed for
future requests, while historical evidence and the provider stop remain intact.
There is no selected pair, five-seed pilot, protocol freeze or M2-R implementation.

## Registered diagnostic conditions

| Condition | Setting |
| --- | --- |
| Environment | Alem `14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e` |
| Goals | collect_wood; collect_stone; make_wood_pickaxe; make_stone_pickaxe |
| Diagnostic seed | 42, independently reset for every candidate/task |
| Episode decision cap | 200; smoke has one decision |
| Sampling | temperature 0; maximum output 2,048 tokens; timeout 120 seconds |
| Actor input | official solo rules, public task, observations, advertised legal actions, eight visible feedback turns |
| Output | native JSON object mode, then common exact one-action/legal-action validation |
| GPT-OSS controls | reasoning_effort medium for both sizes |
| Other controls | provider defaults; exact returned usage and served-model metadata retained |
| Free-only physical cap | 75 calls per provider across cumulative diagnostic scopes; smoke cap 12 |
| Pacing | at least 61 seconds between starts for the same provider |
| Quota uncertainty | one episode per provider per invocation; stop on provider failure; preserve partials |

Provider defaults, tokenizers and reasoning controls are not identical deliberation
budgets. Do not infer usage from response length or add reasoning/cache counts again
to provider totals. API alias and returned served-model identifiers are both recorded.

The original full diagnostic scope was 12 × 4 × 200 = **9,600 requests**. Following
the 404 access exclusion and three additional text candidates, the expanded
registered bound is 14 × 4 × 200 = **11,200 requests**. This is
not authorized as one unattended batch. Scoped plans record episodes × max_steps
and the separate physical allowance before dispatch. Groq's conservative planning
envelope is 200,000 reported tokens and an 8,000-token ceiling per next request;
these are planning guards, not proof of the account's unused quota. Missing usage,
insufficient allowance or a quota failure stops dispatch. A budget-limited prefix
has an unknown capability outcome and never becomes a failed task score. After
three valid Groq smoke responses, the expanded physical allowance increased from
25 to 75 while retaining the token envelope, pacing, prompt and task conditions.
The 8,000-token next-request reserve is a planning assumption, not a measured
tokenizer bound; observed overrun blocks further dispatch.

## Execute a scoped smoke

From `milestones/milestone-1.25`, with credentials supplied only by the process environment:

```sh
uv sync --locked --extra alem --python 3.12
uv run --no-sync python -m mosaic.experiments.run --selection-config ../milestone-1.5/configs/free-model-smoke-expanded-20261003.yaml --preflight
uv run --no-sync python -m mosaic.experiments.run --selection-config ../milestone-1.5/configs/free-model-smoke-expanded-20261003.yaml --confirm-free-tier --candidate openai-gpt-oss-20b --output runs/smoke-20b
```

`--confirm-free-tier` records the operator's already-verified account constraint; it
does not inspect or change billing. For every later scope, supply **all** preceding
session summaries once each using repeated `--prior-summary`. Fatal stopped providers
remain stopped. A nonfatal model-specific 404 persists that model as unavailable
without retry; independent registered models may continue. There is no automatic
retry, re-prompt, fallback or quota recovery.
Smoke pass means a usable provider interaction, not goal success after one action.

## Scientific gates

1. Check access, interface failures and public-prompt parity before capability judgments.
2. Diagnose every available candidate independently on seed 42 across the four goals.
3. Record one explicit pair-selection decision with evidence and rejected candidates.
   Require weak headroom, a materially stronger teacher, and neither shared floor nor ceiling.
   Success ranges 20–50% and 60–90% are planning illustrations, never thresholds.
4. Only then register a new pair-specific pilot: seeds 42, 19, 20, 7 and 11.
   Family selection and any protocol changes must be documented before that pilot.
5. Evaluate paired results, failures, cost dimensions and uncertainty; expand to about
   20 seeds only after reviewing a stable useful gap.

Report per task/model success, steps, calls, accepted/rejected actions, parsing failures,
step-cap saturation, failures/timeouts, usage dimensions and wall time. Exclude
infrastructure-incomplete episodes from capability denominators while retaining costs.
Task labels are CEILING, USEFUL_GAP, SHARED_FLOOR or AMBIGUOUS. One seed cannot establish
population success rates. No useful measured gap means stop skill work and recalibrate
the model pair or task difficulty.

M1.75 and M2-R remain gated by real calibration evidence. M3 remains stopped. The
[Laboratory](../laboratory/README.md) reads saved evidence without controlling execution.
