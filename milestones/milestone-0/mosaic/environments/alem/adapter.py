"""Single-player use of Alem's actual symbolic environment and text wrapper."""

import dataclasses
import json
from importlib.metadata import distribution
from typing import Any, cast

import jax
import numpy as np
from alem.alem_coop.action_masking import compute_action_mask
from alem.alem_coop.constants import Achievement
from alem.alem_coop.constants import Action as AlemAction
from alem.llm.alem_language_wrapper import ACTIONS, AlemLanguageWrapper, make_alem_env
from pydantic import JsonValue

from mosaic.core.config import ALEM_COMMIT
from mosaic.core.models import (
    Action,
    ActionSpec,
    EnvironmentSnapshot,
    Observation,
    StepResult,
    VerificationResult,
)


def _json_value(value: Any) -> JsonValue:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if hasattr(value, "tolist"):
        return cast(JsonValue, value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported Alem state field: {type(value).__name__}")


class AlemEnvironment:
    def __init__(self, agent_id: str = "agent_0", max_steps: int = 20) -> None:
        source = distribution("alem-env").read_text("direct_url.json")
        metadata = json.loads(source) if source else {}
        if metadata.get("vcs_info", {}).get("commit_id") != ALEM_COMMIT:
            raise RuntimeError(
                "Alem must be installed at the pinned commit with uv sync --extra alem"
            )
        self.agent_id = agent_id
        self._initialized = False
        self._env = make_alem_env(
            {
                "ENV_NAME": "Alem-Coop-Symbolic",
                "num_agents": 1,
                "coordination_difficulty": "none",
                "soft_specialization": False,
                "shared_reward": False,
                "max_timesteps": max_steps,
                "specialist_efficiency": 1.0,
                "non_specialist_efficiency": 1.0,
                "randomize_alpha": False,
                "god_mode": False,
            }
        )
        self._wrapper = AlemLanguageWrapper(
            self._env,
            self._env.default_params,
            max_episode_steps=max_steps,
            show_affordances=True,
            render_images=False,
            debug=False,
        )
        self._state: Any = None
        self._rng: Any = None
        self._observations: Any = None

    def _check_agent(self, agent_id: str) -> None:
        if not self._initialized:
            raise RuntimeError("Environment must be reset first")
        if agent_id != self.agent_id:
            raise ValueError(f"Unknown agent: {agent_id}")

    def reset(self, seed: int, task_id: str) -> Observation:
        if task_id != "collect_wood":
            raise ValueError("The initial Alem task is collect_wood")
        self._observations, self._state, self._rng = self._wrapper.reset(jax.random.PRNGKey(seed))
        self._initialized = True
        return self.observe(self.agent_id)

    def observe(self, agent_id: str) -> Observation:
        self._check_agent(agent_id)
        text = self._observations[0]["text"]
        return Observation(
            agent_id=agent_id, text=f"{text['long_term_context']}\n\n{text['short_term_context']}"
        )

    def available_actions(self, agent_id: str) -> tuple[ActionSpec, ...]:
        self._check_agent(agent_id)
        mask = np.asarray(
            compute_action_mask(self._state, self._env.default_params, self._env.static_env_params)
        )[0]
        # The bare Give slot is not a target action and has no meaning with one player.
        return tuple(
            ActionSpec(name=name, description=name)
            for index, name in enumerate(ACTIONS)
            if index < len(mask) and bool(mask[index]) and index != AlemAction.GIVE.value
        )

    def step(self, agent_id: str, action: Action) -> StepResult:
        if action.name not in {spec.name for spec in self.available_actions(agent_id)}:
            return StepResult(
                observation=self.observe(agent_id),
                accepted=False,
                reason="Alem action mask rejected action",
            )
        self._observations, self._state, _, dones, _, self._rng = self._wrapper.step(
            self._state,
            [action.name],
            self._rng,
        )
        time_limit = int(self._state.timestep) >= int(self._env.default_params.max_timesteps)
        return StepResult(
            observation=self.observe(agent_id),
            accepted=True,
            reason="Dispatched to Alem; acceptance does not imply goal progress",
            terminated=bool(dones[0]) and not time_limit,
            truncated=time_limit,
        )

    def verify(self, goal_id: str) -> VerificationResult:
        self._check_agent(self.agent_id)
        if goal_id != "collect_wood":
            raise ValueError("Unsupported Alem goal")
        achieved = bool(self._state.achievements[0, Achievement.COLLECT_WOOD.value])
        return VerificationResult(
            goal_id=goal_id,
            succeeded=achieved,
            evidence=f"state.achievements[0,COLLECT_WOOD]={achieved}",
        )

    def snapshot(self) -> EnvironmentSnapshot:
        self._check_agent(self.agent_id)
        return EnvironmentSnapshot(
            state_json=json.dumps(
                {
                    "state": _json_value(self._state),
                    "rng": _json_value(self._rng),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
