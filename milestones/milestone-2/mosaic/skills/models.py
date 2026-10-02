"""Immutable, data-only procedures and weak-only validation evidence."""

import hashlib
import json
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from mosaic.core.budget import BudgetTotals
from mosaic.core.models import Action, FrozenModel, Observation


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _fields(fields: tuple[str, ...]) -> None:
    if not fields or any(not field.strip() for field in fields):
        raise ValueError("Observation fields must contain nonempty top-level names")
    if len(fields) != len(set(fields)):
        raise ValueError("Observation fields must be unique")


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def project_observation(text: str, fields: tuple[str, ...]) -> str:
    """Project only declared visible JSON fields, without environment-specific mechanics."""
    _fields(fields)
    value = json.loads(text, object_pairs_hook=_object, parse_constant=_nonfinite)
    if not isinstance(value, dict):
        raise ValueError("Skill observations must be JSON objects")
    if any(field not in value for field in fields):
        raise ValueError("Skill observation is missing a declared field")
    return _canonical({field: value[field] for field in fields})


class SkillScope(FrozenModel):
    environment: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    task: str = Field(min_length=1)


class SkillProvenance(FrozenModel):
    source_episode: str = Field(min_length=1)
    source_agent: str = Field(min_length=1)
    source_seed: int = Field(ge=0, le=2**32 - 1)
    source_provider: str = Field(min_length=1)
    source_model: str = Field(min_length=1)
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mosaic_commit: str | None


def candidate_content_sha256(
    scope: SkillScope,
    observation_fields: tuple[str, ...],
    initiation_json: str,
    procedure: tuple[Action, ...],
    allowed_tools: tuple[str, ...],
    termination_goal: str,
) -> str:
    content = {
        "scope": scope.model_dump(mode="json"),
        "observation_fields": observation_fields,
        "initiation_json": initiation_json,
        "procedure": [action.model_dump(mode="json") for action in procedure],
        "allowed_tools": allowed_tools,
        "termination_goal": termination_goal,
    }
    return hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()


class CandidateSkill(FrozenModel):
    skill_id: str = Field(pattern=r"^skill-[0-9a-f]{16}$")
    version: Literal[1] = 1
    status: Literal["candidate"] = "candidate"
    scope: SkillScope
    observation_fields: tuple[str, ...]
    initiation_json: str
    procedure: tuple[Action, ...] = Field(min_length=1, max_length=256)
    allowed_tools: tuple[str, ...]
    termination_goal: str
    provenance: SkillProvenance
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("observation_fields")
    @classmethod
    def unique_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _fields(value)
        return value

    @model_validator(mode="after")
    def valid_content(self) -> Self:
        projected = project_observation(self.initiation_json, self.observation_fields)
        if projected != self.initiation_json:
            raise ValueError("Initiation must be canonical JSON with exactly the declared fields")
        if len(self.allowed_tools) != len(set(self.allowed_tools)) or set(self.allowed_tools) != {
            action.name for action in self.procedure
        }:
            raise ValueError("Allowed tools must be the unique exact procedure action names")
        if self.termination_goal != self.scope.task:
            raise ValueError("Termination goal must match the scoped task")
        digest = candidate_content_sha256(
            self.scope,
            self.observation_fields,
            self.initiation_json,
            self.procedure,
            self.allowed_tools,
            self.termination_goal,
        )
        if self.content_sha256 != digest or self.skill_id != f"skill-{digest[:16]}":
            raise ValueError("Candidate identity and digest must match canonical content")
        return self

    def matches(self, scope: SkillScope, observation: Observation) -> bool:
        if scope != self.scope:
            return False
        try:
            return (
                project_observation(observation.text, self.observation_fields)
                == self.initiation_json
            )
        except (TypeError, ValueError, OverflowError, RecursionError):
            return False


class ValidationTrial(FrozenModel):
    seed: int = Field(ge=0, le=2**32 - 1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    status: Literal["succeeded", "failed", "error"]
    reason: str
    skill_completed: bool
    budget: BudgetTotals

    @model_validator(mode="after")
    def weak_only(self) -> Self:
        teacher = self.budget.teacher
        if (
            teacher.model_calls
            or teacher.input_tokens not in (None, 0)
            or teacher.output_tokens not in (None, 0)
            or teacher.known_input_tokens
            or teacher.known_output_tokens
            or teacher.unknown_input_calls
            or teacher.unknown_output_calls
            or teacher.synthetic_usage_calls
            or teacher.wall_clock_ms
        ):
            raise ValueError("Skill validation must not use the teacher")
        return self


class ValidationReport(FrozenModel):
    skill_id: str = Field(pattern=r"^skill-[0-9a-f]{16}$")
    version: int = Field(ge=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trials: tuple[ValidationTrial, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_seeds(self) -> Self:
        if len(self.trials) != len({trial.seed for trial in self.trials}):
            raise ValueError("Validation seeds must be unique")
        return self

    @property
    def passed(self) -> bool:
        return all(
            trial.status == "succeeded"
            and trial.skill_completed
            and trial.budget.teacher.model_calls == 0
            for trial in self.trials
        )


class VerifiedSkill(FrozenModel):
    status: Literal["verified"] = "verified"
    candidate: CandidateSkill
    validation: ValidationReport

    @model_validator(mode="after")
    def valid_validation(self) -> Self:
        if (
            self.candidate.skill_id != self.validation.skill_id
            or self.candidate.version != self.validation.version
            or self.candidate.content_sha256 != self.validation.content_sha256
        ):
            raise ValueError("Validation report must match candidate identity, version and content")
        if not self.validation.passed:
            raise ValueError("Only candidates passing every weak-only validation may be promoted")
        return self
