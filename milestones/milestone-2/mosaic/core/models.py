"""Small, deeply immutable domain values shared by adapters and the runner."""

import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ModelRole = Literal["weak", "teacher"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, strict=True)


class Observation(FrozenModel):
    agent_id: str
    text: str


class ActionSpec(FrozenModel):
    name: str = Field(min_length=1)
    description: str


class Action(FrozenModel):
    name: str = Field(min_length=1)


class StepResult(FrozenModel):
    observation: Observation
    accepted: bool
    reason: str
    terminated: bool = False
    truncated: bool = False


class VerificationResult(FrozenModel):
    goal_id: str
    succeeded: bool
    evidence: str


class EnvironmentSnapshot(FrozenModel):
    # Canonical JSON text permits adapter-specific state without mutable payloads.
    state_json: str

    @field_validator("state_json")
    @classmethod
    def valid_json(cls, value: str) -> str:
        json.loads(value)
        return value


class TokenUsage(FrozenModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    measurement: Literal["provider", "synthetic", "unavailable"] = "unavailable"

    @model_validator(mode="after")
    def consistent_measurement(self) -> Self:
        counts = (self.input_tokens, self.output_tokens)
        if self.measurement == "unavailable" and any(value is not None for value in counts):
            raise ValueError("Known usage must identify its measurement source")
        if self.measurement == "synthetic" and any(value is None for value in counts):
            raise ValueError("Synthetic fixture usage must include both counts")
        return self


class AttemptFeedback(FrozenModel):
    step: int = Field(ge=1)
    model_role: ModelRole
    proposal: str
    action: Action | None
    accepted: bool
    feedback: str
    observation: Observation


class SkillHint(FrozenModel):
    skill_id: str
    version: int = Field(ge=1)
    content_sha256: str
    procedure: tuple[Action, ...] = Field(min_length=1, max_length=256)
    cursor: int = Field(ge=0)

    @model_validator(mode="after")
    def bounded_cursor(self) -> Self:
        if self.cursor >= len(self.procedure):
            raise ValueError("Skill cursor must name a remaining procedure action")
        return self


class ModelRequest(FrozenModel):
    model: str
    system_prompt: str
    observation: Observation
    available_actions: tuple[ActionSpec, ...]
    temperature: float = Field(ge=0, le=2)
    max_output_tokens: int = Field(gt=0)
    model_role: ModelRole = "weak"
    goal: str = ""
    step: int = Field(default=0, ge=0)
    history: tuple[AttemptFeedback, ...] = ()
    skill: SkillHint | None = None


class ModelResponse(FrozenModel):
    provider: str
    model: str
    text: str
    usage: TokenUsage
    finish_reason: str
