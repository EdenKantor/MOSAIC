# Alem calibration prompt and upstream comparison

This calibration adapter uses Alem at commit
`14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e`. It gives real weak and teacher models the same
grounded game rules and actor-visible information before interpreting performance. It does
not reproduce Alem's official cooperative benchmark, and no scores are inferred from this
source inspection.

The inspected sources are the pinned [baseline configuration](https://github.com/alem-world/alem-env/blob/14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e/baselines/llm/config/config.yaml),
[prompt builder](https://github.com/alem-world/alem-env/blob/14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e/baselines/llm/eval_utils/prompt_builder.py),
[naive agent](https://github.com/alem-world/alem-env/blob/14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e/baselines/llm/eval_utils/agents/naive.py),
[base agent](https://github.com/alem-world/alem-env/blob/14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e/baselines/llm/eval_utils/agents/base.py),
[chain-of-thought agent](https://github.com/alem-world/alem-env/blob/14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e/baselines/llm/eval_utils/agents/chain_of_thought.py),
[configured robust agent](https://github.com/alem-world/alem-env/blob/14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e/baselines/llm/eval_utils/agents/robust_all.py)
and [evaluator](https://github.com/alem-world/alem-env/blob/14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e/baselines/llm/eval_utils/evaluator.py).
Installed pinned sources also confirm the [official environment wrapper](https://github.com/alem-world/alem-env/blob/14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e/alem/llm/alem_env.py),
[solo wrapper and rules](https://github.com/alem-world/alem-env/blob/14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e/alem/llm/alem_language_wrapper_single.py)
and [action-mask implementation](https://github.com/alem-world/alem-env/blob/14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e/alem/alem_coop/action_masking.py).

## Why the earlier fixture prompt is insufficient for calibration

M0–M2 use a minimal action-selection instruction and an exact JSON action contract. The Alem
adapter supplies observations and available action names, but those milestones do not inject
the official game-rule prompt. Their fake policies know how to act from fixture logic. That
is sufficient for engineering checks and does not establish what a real model can do.

For a real model, missing movement/facing semantics, interaction rules, crafting prerequisites
and survival mechanics could artificially depress both weak and teacher performance. Weak-only
episodes without routing also omit actor history in the earlier runner. Poor results under
that interface would confound model capability with scaffolding quality. No real-model result
is retroactively claimed or corrected by this observation.

`AlemCalibrationEnvironment.instruction_prompt()` instead calls the official
`get_instruction_prompt_single()` with `prompt_mode="specific"`, level-based disclosure and
the declared public focused task. The parent agent adds the same one-action JSON output
contract for both model roles. Rules come from upstream rather than a fabricated task solver.

The API-only candidate protocol added on 2026-10-03 serializes identical system/user
content for Gemini and Groq. Gemini maps it into `systemInstruction`/`contents`; Groq uses
chat messages. Both request native JSON object mode without an action enum or schema
constraint, leaving malformed shape and action legality to the same local validator.
Framework role labels and agent IDs remain absent from current and history input.
Gemini's `includeThoughts=false` omits returned thought text; it does not establish that
internal reasoning is disabled. Provider-reported thought usage remains a separate dimension.

## Rules, observations and action access

The calibration uses `AlemLanguageWrapperSingle`, as the official `CraftaxEnv` does when its
player count is one. It removes teammate and coordination instruction sections. The specific
rules explain movement and facing, `Do`, placing, crafting, resource/tool progression,
survival and the applicable dungeon mechanics. Later-level sections are gated using the
player level already shown in visible status. The official evaluator keeps rules for the
highest level visited; this adapter's prompt method uses the current visible level. The
focused calibration tasks are surface resource/tool tasks.

| Setting | Pinned baseline configuration | MOSAIC calibration |
| --- | --- | --- |
| Prompt mode | `specific_collaborative`; solo wrapper demotes it to `specific` | `specific` through the official solo builder |
| Location precision | `precise_location=true`, `exact_coordinates=true`, `egocentric=false` | Same |
| Visible-item formatting | Unique items; omit grass/sand/path; water at region edges | Same |
| Rule disclosure | `progressively_display_game_info=true` | `progressive_disclosure=true`, current visible level |
| Full action block | `include_all_actions=false` | Same; legal actions supplied each turn |
| Affordances in text | `show_affordances=true` | Same, with the solo-only correction below |
| Model images | `max_image_history=0`, ASCII/image-scene disabled | No images, ASCII or image-scene |
| Rendering/debug | Configuration enables debug/rendered audit artifacts | Disabled in calibration |

The inherited solo affordance formatter in this pin still enumerates the cooperative action
list, including Request actions. A calibration-only subclass filters those actor-visible
affordances through `SINGLE_AGENT_ACTIONS`; Request/Give are also removed from MOSAIC's
advertised actions. Both lists use the official mask and original cooperative indices.
Enumerating the shorter solo list against the original mask would misalign indices. This
format correction changes no simulator transition or mask rule.

Upstream's wrapper derives affordance text from the official mask. Its action parser accepts
canonical/fallback matches, and the wrapper can still dispatch an action whose mask is false,
then report ineffective action feedback. MOSAIC requires exact JSON and membership in its
advertised masked list before dispatch. Movement, `Do`, Sleep and Rest remain conservatively
available under the official mask; availability does not prove useful progress. The calibration
exposes descriptions from the official solo action dictionary as well as legal action names.
These stricter parsing/dispatch semantics are a declared difference from upstream.

## Reachable single-agent task settings

The legacy one-player smoke sets `soft_specialization=false`. Upstream world generation assigns
player zero the warrior role. The official mask and crafting logic then hard-gate wood and
stone pickaxes to miners. Efficiencies of 1.0 alone do not disable that gate. Preserving this
combination would make the pickaxe tasks inaccessible to the solo warrior; a failed score
would not diagnose model competence.

The new calibration therefore sets `soft_specialization=true`, with specialist and
non-specialist efficiencies both 1.0, for both model roles. This unblocks the resource/tool
abilities required by the registered tasks without adding items, changing world seeds or
granting god mode. M0–M2 adapters remain unchanged. The pinned cooperative benchmark also
uses soft specialization, but its non-specialist efficiency is 0.2 rather than 1.0.

Other calibration world settings are explicit and identical across roles: one player,
coordination `none`, shared reward false, randomized efficiency false and god mode false.
The configured decision cap also sets the simulator `max_timesteps` and wrapper episode cap.
Upstream's default comparison instead has three players, coordination `easy`, and a 10,000-step
cap. This solo calibration is a different environment condition and outcome definition.

The supported goals are `collect_wood`, `collect_stone`, `make_wood_pickaxe` and
`make_stone_pickaxe`. Success reads the corresponding official `Achievement` bit. The prompt
contains the public task description, not the private verifier evidence or audit snapshot.
Observations remain the wrapper's visible long-term and short-term text. Complete state and
PRNG snapshots remain audit data and are not inserted into actor requests.

## History, reasoning and output protocol

The baseline prompt builder stores alternating observations and actions, includes up to eight
recent text observations under the pinned config, and appends short-term inventory/status
context only to the latest observation. It can retain prior reasoning, scratchpads and
communication when the selected agent supplies them. Naive and chain-of-thought agents both
feed back the previous action; the latter also asks for step-by-step reasoning and stores
its generated plan in history.

Calibration supplies current visible text and a configured history bound of eight visible
turn-feedback entries for both weak and teacher episodes. Entries retain proposals, actions,
acceptance/rejection feedback and visible post-action observations. This is bounded history,
not an identical serialization of upstream's alternating chat messages. No cross-episode
history or private state is shared.

The pinned config selects `robust_all`, enables reasoning/scratchpad/communication features,
and uses tagged action output with robust fallback extraction. Its base client proxy supports
parse-feedback retry calls; the pinned `max_parse_retries` is zero, while client transport
retry configuration is five. The evaluator can issue post-episode model debriefs. Calibration
uses one strict JSON action per decision, no requested chain-of-thought, reflection,
scratchpad, communication, parser retry or hidden transport retry. Rejected proposals and
provider failures remain recorded attempts rather than repaired outcomes.

Model-specific thinking controls must be recorded explicitly. A local chat-template thinking
flag and a hosted provider's reasoning-effort parameter are different mechanisms. The pilot
enables local thinking and hosted medium reasoning. Identical environment information and output contracts
do not make those controls equivalent. Reasoning usage, when reported, remains part of raw
inference accounting.

The calibration configuration is in
[milestone-1.5/configs/calibration-pilot.yaml](../../milestone-1.5/configs/calibration-pilot.yaml).
Its purpose is to establish useful weak/teacher behavior before protocol freeze and real-trajectory
skill redesign. This inspection is not evidence of useful transfer, an official benchmark
score or teacher-withdrawal retention. Executed outcomes belong in the calibration report.
