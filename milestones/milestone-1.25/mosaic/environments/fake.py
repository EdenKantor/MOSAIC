"""Seeded test fixture: fetch a spare part and return to repair a pump."""

import json
import random

from mosaic.core.models import (
    Action,
    ActionSpec,
    EnvironmentSnapshot,
    Observation,
    StepResult,
    VerificationResult,
)


class FakeEnvironment:
    def __init__(self, agent_id: str = "agent_0") -> None:
        self.agent_id = agent_id
        self._initialized = False
        self._position = 0
        self._part_position = 0
        self._has_part = False
        self._repaired = False
        self._steps = 0

    def reset(self, seed: int, task_id: str) -> Observation:
        if task_id != "repair_pump":
            raise ValueError(f"Unsupported fake task: {task_id}")
        self._position = 0
        self._part_position = random.Random(seed).randint(2, 8)
        self._has_part = False
        self._repaired = False
        self._steps = 0
        self._initialized = True
        return self.observe(self.agent_id)

    def _check_agent(self, agent_id: str) -> None:
        if not self._initialized:
            raise RuntimeError("Environment must be reset first")
        if agent_id != self.agent_id:
            raise ValueError(f"Unknown agent: {agent_id}")

    def observe(self, agent_id: str) -> Observation:
        self._check_agent(agent_id)
        return Observation(agent_id=agent_id, text=self.snapshot().state_json)

    def available_actions(self, agent_id: str) -> tuple[ActionSpec, ...]:
        self._check_agent(agent_id)
        if self._repaired:
            return ()
        actions = [ActionSpec(name="wait", description="Stay in place")]
        if self._position < self._part_position:
            actions.append(ActionSpec(name="move_east", description="Move one tile east"))
        if self._position > 0:
            actions.append(ActionSpec(name="move_west", description="Move one tile west"))
        if self._position == self._part_position and not self._has_part:
            actions.append(ActionSpec(name="collect", description="Collect the spare part"))
        if self._position == 0 and self._has_part:
            actions.append(
                ActionSpec(name="repair", description="Use the spare to repair the pump")
            )
        return tuple(actions)

    def step(self, agent_id: str, action: Action) -> StepResult:
        allowed = {spec.name for spec in self.available_actions(agent_id)}
        if action.name not in allowed:
            return StepResult(
                observation=self.observe(agent_id),
                accepted=False,
                reason="Action preconditions are not satisfied",
                terminated=self._repaired,
            )
        if action.name == "move_east":
            self._position += 1
        elif action.name == "move_west":
            self._position -= 1
        elif action.name == "collect":
            self._has_part = True
        elif action.name == "repair":
            self._repaired = True
            self._has_part = False
        self._steps += 1
        return StepResult(
            observation=self.observe(agent_id),
            accepted=True,
            reason="Applied",
            terminated=self._repaired,
        )

    def verify(self, goal_id: str) -> VerificationResult:
        self._check_agent(self.agent_id)
        if goal_id != "repair_pump":
            raise ValueError(f"Unsupported goal: {goal_id}")
        return VerificationResult(
            goal_id=goal_id,
            succeeded=self._repaired,
            evidence=f"pump_repaired={self._repaired}",
        )

    def snapshot(self) -> EnvironmentSnapshot:
        self._check_agent(self.agent_id)
        return EnvironmentSnapshot(
            state_json=json.dumps(
                {
                    "position": self._position,
                    "part_position": self._part_position,
                    "has_part": self._has_part,
                    "pump_repaired": self._repaired,
                    "steps": self._steps,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
