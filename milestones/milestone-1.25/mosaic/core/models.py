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
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    provider_total_tokens: int | None = Field(default=None, ge=0)
    provider_request_id: str | None = Field(default=None, min_length=1)
    measurement: Literal["provider", "synthetic", "unavailable"] = "unavailable"
    measurement_source: str | None = Field(default=None, min_length=1)

    @field_validator("provider_request_id", "measurement_source")
    @classmethod
    def nonempty_metadata(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Usage metadata must not be blank")
        return value

    @model_validator(mode="after")
    def consistent_measurement(self) -> Self:
        counts = (
            self.input_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.cached_input_tokens,
            self.provider_total_tokens,
        )
        if self.measurement == "unavailable" and any(value is not None for value in counts):
            raise ValueError("Known usage must identify its measurement source")
        if self.measurement == "synthetic" and (
            self.input_tokens is None or self.output_tokens is None
        ):
            raise ValueError("Synthetic fixture usage must include both counts")
        if (
            self.input_tokens is not None
            and self.cached_input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise ValueError("Cached input usage cannot exceed reported input usage")
        return self


class AttemptFeedback(FrozenModel):
    step: int = Field(ge=1)
    model_role: ModelRole
    proposal: str
    action: Action | None
    accepted: bool
    feedback: str
    observation: Observation


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
    thinking: bool | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    context_window: int | None = Field(default=None, gt=0)
    thinking_budget: int | None = Field(default=None, ge=0)
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None


class ProviderRateLimits(FrozenModel):
    request_limit: int | None = Field(default=None, ge=0)
    token_limit: int | None = Field(default=None, ge=0)
    remaining_requests: int | None = Field(default=None, ge=0)
    remaining_tokens: int | None = Field(default=None, ge=0)
    request_reset_seconds: float | None = Field(default=None, ge=0)
    token_reset_seconds: float | None = Field(default=None, ge=0)


class ModelResponse(FrozenModel):
    provider: str
    model: str
    text: str
    usage: TokenUsage
    finish_reason: str
    reported_model: str | None = None
    rate_limits: ProviderRateLimits | None = None
