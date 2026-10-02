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
    provider: Literal["fake", "ollama", "groq"]
    model: str = Field(min_length=1)
    policy: Literal["repair_pump", "first_available", "wait_then_repair"] = "repair_pump"
    failed_calls: int = Field(default=0, ge=0)
    temperature: float = Field(default=0, ge=0, le=2)
    max_output_tokens: int = Field(default=64, gt=0)
    thinking: bool | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    context_window: int | None = Field(default=None, gt=0)
    timeout_seconds: float = Field(default=120, gt=0)

    @model_validator(mode="after")
    def deterministic_sampling(self) -> Self:
        if self.provider == "fake" and self.temperature != 0:
            raise ValueError("The deterministic fake provider requires temperature=0")
        if self.provider != "ollama" and (
            self.thinking is not None or self.context_window is not None
        ):
            raise ValueError("Thinking/context-window controls apply only to the Ollama adapter")
        if self.provider != "groq" and self.reasoning_effort is not None:
            raise ValueError("Reasoning-effort controls apply only to the Groq adapter")
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
    prompt_protocol: Literal["minimal", "alem_official_v1"] = "minimal"
    history_limit: int = Field(default=8, ge=0)
    agent_model_role: Literal["weak", "teacher"] = "weak"

    @model_validator(mode="after")
    def supported_task(self) -> Self:
        supported = (
            ("repair_pump",)
            if self.environment.name == "fake"
            else ("collect_wood", "collect_stone", "make_wood_pickaxe", "make_stone_pickaxe")
        )
        if self.task not in supported:
            raise ValueError("Task is not in this adapter's inspected grounded task registry")
        if (
            self.environment.name == "alem"
            and self.model.provider == "fake"
            and self.model.policy != "first_available"
        ):
            raise ValueError(
                "Alem smoke requires the environment-independent first_available policy"
            )
        if (self.teacher is None) != (self.routing is None):
            raise ValueError("Teacher and routing configuration must be supplied together")
        if (
            self.teacher
            and self.environment.name == "alem"
            and self.teacher.model.provider == "fake"
            and self.teacher.model.policy != "first_available"
        ):
            raise ValueError("The scripted Alem teacher must use first_available")
        if self.prompt_protocol == "alem_official_v1" and self.environment.name != "alem":
            raise ValueError("The official Alem prompt requires the Alem environment")
        if self.model.provider != "fake" and self.prompt_protocol != "alem_official_v1":
            raise ValueError("Real-model experiments require the reviewed Alem prompt protocol")
        if self.agent_model_role == "teacher" and self.routing is not None:
            raise ValueError("An independent teacher arm cannot also use escalation routing")
        return self


def load_config(path: Path) -> ExperimentConfig:
    # JSON validation accepts enum string values while keeping scalar types strict.
    return ExperimentConfig.model_validate_json(
        json.dumps(yaml.safe_load(path.read_text(encoding="utf-8")))
    )


class CalibrationFamily(FrozenModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    task: str = Field(min_length=1)
    max_steps: int = Field(gt=0)


class CalibrationConfig(FrozenModel):
    """Independent paired actors; escalation and procedural memory are absent."""

    calibration_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    stage: Literal["pilot", "expanded"] = "pilot"
    environment: EnvironmentConfig
    agent: AgentConfig
    weak: ModelConfig
    teacher: ModelConfig
    families: tuple[CalibrationFamily, ...] = Field(min_length=1, max_length=5)
    seeds: tuple[int, ...] = Field(min_length=1)
    prompt_protocol: Literal["minimal", "alem_official_v1"] = "alem_official_v1"
    history_limit: int = Field(default=8, ge=0)

    @model_validator(mode="after")
    def paired_protocol(self) -> Self:
        if len(set(self.seeds)) != len(self.seeds) or any(
            seed < 0 or seed > 2**32 - 1 for seed in self.seeds
        ):
            raise ValueError("Paired seeds must be unique unsigned 32-bit values")
        if len({family.name for family in self.families}) != len(self.families):
            raise ValueError("Task-family names must be unique")
        if (self.weak.temperature, self.weak.max_output_tokens) != (
            self.teacher.temperature,
            self.teacher.max_output_tokens,
        ):
            raise ValueError("Paired actors require the same sampling temperature and output cap")
        if self.environment.name == "fake":
            if self.weak.provider != "fake" or self.teacher.provider != "fake":
                raise ValueError("Fake calibration fixtures cannot dispatch real providers")
            if self.prompt_protocol != "minimal":
                raise ValueError("Fake fixtures use the minimal prompt protocol")
        else:
            if len({family.task for family in self.families}) != len(self.families):
                raise ValueError("Real calibration task families must have distinct grounded goals")
            if (self.weak.provider, self.weak.model) == (self.teacher.provider, self.teacher.model):
                raise ValueError("Real capability calibration needs distinct candidate identities")
            if (
                self.weak.provider == "fake"
                or self.teacher.provider == "fake"
                or self.prompt_protocol != "alem_official_v1"
            ):
                raise ValueError(
                    "Real calibration requires two real providers and the reviewed prompt"
                )
            if len(self.families) < 3:
                raise ValueError("Real calibration pilots need three to five task families")
        for family in self.families:
            # Reuse the immutable episode task/protocol checks without creating a teacher gate.
            ExperimentConfig(
                experiment_id=f"{self.calibration_id}-{family.name}",
                seed=self.seeds[0],
                environment=self.environment,
                task=family.task,
                agent=self.agent,
                model=self.weak,
                max_steps=family.max_steps,
                prompt_protocol=self.prompt_protocol,
                history_limit=self.history_limit,
            )
        return self


def load_calibration_config(path: Path) -> CalibrationConfig:
    return CalibrationConfig.model_validate_json(
        json.dumps(yaml.safe_load(path.read_text(encoding="utf-8")))
    )
