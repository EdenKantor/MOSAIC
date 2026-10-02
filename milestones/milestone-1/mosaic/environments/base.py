from typing import Protocol

from mosaic.core.models import (
    Action,
    ActionSpec,
    EnvironmentSnapshot,
    Observation,
    StepResult,
    VerificationResult,
)


class EnvironmentAdapter(Protocol):
    def reset(self, seed: int, task_id: str) -> Observation: ...

    def observe(self, agent_id: str) -> Observation: ...

    def available_actions(self, agent_id: str) -> tuple[ActionSpec, ...]: ...

    def step(self, agent_id: str, action: Action) -> StepResult: ...

    def verify(self, goal_id: str) -> VerificationResult: ...

    def snapshot(self) -> EnvironmentSnapshot: ...
