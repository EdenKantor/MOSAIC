import json
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from mosaic.core.models import FrozenModel

ALEM_COMMIT = "14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e"


class TeacherAccess(StrEnum):
    AVAILABLE = "available"
    WITHDRAWN = "withdrawn"


class EnvironmentConfig(FrozenModel):
    name: Literal["fake", "alem"]
    revision: str

    @model_validator(mode="after")
    def check_revision(self) -> Self:
        expected = "1" if self.name == "fake" else ALEM_COMMIT
        if self.revision != expected:
            raise ValueError(f"{self.name} requires inspected revision {expected}")
        return self


class AgentConfig(FrozenModel):
    agent_id: str = Field(min_length=1)


class ModelConfig(FrozenModel):
    provider: Literal["fake"]
    model: str = Field(min_length=1)
    policy: Literal["repair_pump", "first_available", "wait_then_repair", "follow_skill"] = (
        "repair_pump"
    )
    failed_calls: int = Field(default=0, ge=0)
    temperature: float = Field(default=0, ge=0, le=2)
    max_output_tokens: int = Field(default=64, gt=0)

    @model_validator(mode="after")
    def deterministic_sampling(self) -> Self:
        if self.temperature != 0:
            raise ValueError("The deterministic fake provider requires temperature=0")
        return self


class TeacherConfig(FrozenModel):
    access: TeacherAccess = TeacherAccess.AVAILABLE
    model: ModelConfig


class RoutingConfig(FrozenModel):
    strategy: Literal["bounded_failure"] = "bounded_failure"
    attempt_steps: int = Field(default=8, gt=0)
    recovery_attempts: int = Field(default=1, ge=0)
    recovery_steps: int = Field(default=8, gt=0)
    consecutive_failure_limit: int = Field(default=2, gt=0)
    history_limit: int = Field(default=8, gt=0)


class SkillUseConfig(FrozenModel):
    mode: Literal["validation", "reuse"]
    playbook_path: str | None = None
    candidate_id: str | None = None

    @model_validator(mode="after")
    def explicit_source(self) -> Self:
        if self.mode == "reuse":
            if not self.playbook_path or self.candidate_id is not None:
                raise ValueError("Reuse requires a PlayBook path and no candidate override")
        elif not self.candidate_id or self.playbook_path is not None:
            raise ValueError("Validation requires a candidate ID and no PlayBook")
        return self


class ExperimentConfig(FrozenModel):
    experiment_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    seed: int = Field(ge=0, le=2**32 - 1)
    environment: EnvironmentConfig
    task: str = Field(min_length=1)
    agent: AgentConfig
    model: ModelConfig
    max_steps: int = Field(gt=0)
    teacher: TeacherConfig | None = None
    routing: RoutingConfig | None = None
    skills: SkillUseConfig | None = None

    @model_validator(mode="after")
    def supported_task(self) -> Self:
        expected = "repair_pump" if self.environment.name == "fake" else "collect_wood"
        if self.task != expected:
            raise ValueError(f"{self.environment.name} currently supports task {expected}")
        if self.environment.name == "alem" and self.model.policy != "first_available":
            raise ValueError(
                "Alem smoke requires the environment-independent first_available policy"
            )
        if (self.teacher is None) != (self.routing is None):
            raise ValueError("Teacher and routing configuration must be supplied together")
        if (
            self.teacher
            and self.environment.name == "alem"
            and self.teacher.model.policy != "first_available"
        ):
            raise ValueError("The scripted Alem teacher must use first_available")
        if self.skills and self.skills.mode == "validation" and self.teacher is not None:
            raise ValueError("Skill validation must be weak-only with no teacher or routing")
        return self


class CompilationConfig(FrozenModel):
    observation_fields: tuple[str, ...] = Field(min_length=1)
    max_actions: int = Field(default=32, gt=0, le=256)

    @model_validator(mode="after")
    def distinct_fields(self) -> Self:
        if any(not field for field in self.observation_fields) or len(
            set(self.observation_fields)
        ) != len(self.observation_fields):
            raise ValueError("Compilation needs unique nonempty visible observation fields")
        return self


class ValidationConfig(FrozenModel):
    seeds: tuple[int, ...] = Field(min_length=1)
    max_steps: int = Field(default=32, gt=0)

    @model_validator(mode="after")
    def distinct_seeds(self) -> Self:
        if len(set(self.seeds)) != len(self.seeds) or any(
            not 0 <= seed <= 2**32 - 1 for seed in self.seeds
        ):
            raise ValueError("Seeds must be unique uint32 values")
        return self


class ReuseConfig(ValidationConfig):
    agent_id: str = Field(default="agent_1", min_length=1)


class LearningConfig(FrozenModel):
    experiment_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    acquisition: ExperimentConfig
    compilation: CompilationConfig
    validation: ValidationConfig
    reuse: ReuseConfig

    @model_validator(mode="after")
    def teacher_acquisition(self) -> Self:
        if self.acquisition.teacher is None or self.acquisition.skills is not None:
            raise ValueError("Acquisition requires routing and no preexisting skill injection")
        return self


def load_config(path: Path) -> ExperimentConfig:
    # JSON validation accepts enum string values while keeping scalar types strict.
    return ExperimentConfig.model_validate_json(
        json.dumps(yaml.safe_load(path.read_text(encoding="utf-8")))
    )


def load_learning_config(path: Path) -> LearningConfig:
    return LearningConfig.model_validate_json(
        json.dumps(yaml.safe_load(path.read_text(encoding="utf-8")))
    )
