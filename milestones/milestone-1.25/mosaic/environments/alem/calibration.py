"""Grounded single-agent Alem observations and rules for real-model calibration."""

from typing import Any

import jax
import numpy as np
from alem.alem_coop.action_masking import compute_action_mask
from alem.alem_coop.constants import Achievement
from alem.llm.alem_language_wrapper import ACTIONS
from alem.llm.alem_language_wrapper_single import (
    SINGLE_AGENT_ACTION_DICT,
    SINGLE_AGENT_ACTIONS,
    AlemLanguageWrapperSingle,
    get_instruction_prompt_single,
)

from mosaic.core.models import ActionSpec, Observation, VerificationResult
from mosaic.environments.alem.adapter import AlemEnvironment

_TASK_ACHIEVEMENTS = {
    "collect_wood": Achievement.COLLECT_WOOD,
    "collect_stone": Achievement.COLLECT_STONE,
    "make_wood_pickaxe": Achievement.MAKE_WOOD_PICKAXE,
    "make_stone_pickaxe": Achievement.MAKE_STONE_PICKAXE,
}
_TASK_DESCRIPTIONS = {
    "collect_wood": "Collect wood while staying alive.",
    "collect_stone": "Collect stone while staying alive.",
    "make_wood_pickaxe": "Craft a wood pickaxe while staying alive.",
    "make_stone_pickaxe": "Craft a stone pickaxe while staying alive.",
}
_SINGLE_ACTION_NAMES = frozenset(SINGLE_AGENT_ACTIONS)


# The optional upstream wrapper has no typing metadata in the core-only environment.
class _CalibrationLanguageWrapper(AlemLanguageWrapperSingle):  # type: ignore[misc]
    """Keep solo affordances consistent with the solo action list.

    The pinned solo wrapper inherits a cooperative affordance formatter that still
    lists Request actions. Filtering that actor-visible list does not change the
    simulator mask or expose additional information.
    """

    def get_affordances(self, state: Any, player_idx: int) -> str:
        mask = np.asarray(compute_action_mask(state, self.env_params, self.static_env_params))[
            player_idx
        ]
        names = tuple(
            name
            for index, name in enumerate(ACTIONS)
            if name in _SINGLE_ACTION_NAMES and index < len(mask) and bool(mask[index])
        )
        return "Available actions:\n - " + "\n - ".join(names) if names else ""


class AlemCalibrationEnvironment(AlemEnvironment):
    """Keep the pinned simulator, but use its official solo wrapper and full rules.

    Inherited execution, visible observation formatting and audit snapshots keep the
    same adapter boundary. The original cooperative action indices must be preserved
    when masking, even though solo-only names are filtered out of the actor interface.
    """

    def __init__(self, agent_id: str = "agent_0", max_steps: int = 20) -> None:
        if not agent_id.strip():
            raise ValueError("Agent ID must not be blank")
        if max_steps <= 0:
            raise ValueError("Alem calibration needs a positive decision cap")
        # The base constructor checks installed VCS provenance and fixes identical
        # single-player simulator settings for every weak/teacher calibration arm.
        super().__init__(agent_id=agent_id, max_steps=max_steps, soft_specialization=True)
        self._wrapper = _CalibrationLanguageWrapper(
            self._env,
            self._env.default_params,
            max_episode_steps=max_steps,
            unique_items=True,
            precise_location=True,
            exact_coordinates=True,
            egocentric=False,
            skip_items=["grass", "sand", "path"],
            edge_only_items=["water"],
            llm_mode="easy",
            prompt_mode="specific",
            show_affordances=True,
            render_images=False,
            debug=False,
            use_ascii=False,
            use_image_scene=False,
        )
        self._task_id: str | None = None

    def reset(self, seed: int, task_id: str) -> Observation:
        if task_id not in _TASK_ACHIEVEMENTS:
            raise ValueError(f"Unsupported Alem calibration task: {task_id}")
        self._observations, self._state, self._rng = self._wrapper.reset(jax.random.PRNGKey(seed))
        self._task_id = task_id
        self._initialized = True
        return self.observe(self.agent_id)

    def available_actions(self, agent_id: str) -> tuple[ActionSpec, ...]:
        self._check_agent(agent_id)
        mask = np.asarray(
            compute_action_mask(self._state, self._env.default_params, self._env.static_env_params)
        )[0]
        # SINGLE_AGENT_ACTIONS removes entries from ACTIONS. Enumerating that shorter
        # list would misalign action-mask indices; filter the original indexed list.
        return tuple(
            ActionSpec(name=name, description=SINGLE_AGENT_ACTION_DICT[name])
            for index, name in enumerate(ACTIONS)
            if name in _SINGLE_ACTION_NAMES and index < len(mask) and bool(mask[index])
        )

    def verify(self, goal_id: str) -> VerificationResult:
        self._check_agent(self.agent_id)
        achievement = _TASK_ACHIEVEMENTS.get(goal_id)
        if achievement is None:
            raise ValueError(f"Unsupported Alem calibration goal: {goal_id}")
        achieved = bool(self._state.achievements[0, achievement.value])
        return VerificationResult(
            goal_id=goal_id,
            succeeded=achieved,
            evidence=f"state.achievements[0,{achievement.name}]={achieved}",
        )

    def instruction_prompt(self) -> str:
        self._check_agent(self.agent_id)
        assert self._task_id is not None
        rules = get_instruction_prompt_single(
            prompt_mode="specific",
            include_all_actions=False,
            progressive_disclosure=True,
            llm_mode="easy",
            # The wrapper also exposes this level in its actor-visible status.
            current_level=int(self._state.player_level),
            role="warrior",
            agent_id=0,
        )
        # This is the declared public task, not verifier evidence or hidden state.
        return (
            f"{rules}\n\n## Focused calibration task\n"
            f"{_TASK_DESCRIPTIONS[self._task_id]} "
            "Prioritize this task rather than maximizing the overall achievement score."
        )
