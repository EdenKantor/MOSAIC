from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from mosaic.core.models import FrozenModel

ALEM_COMMIT = "14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e"


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
    model: Literal["deterministic-policy-v1"]
    policy: Literal["repair_pump", "first_available"] = "repair_pump"
    temperature: float = Field(default=0, ge=0, le=2)
    max_output_tokens: int = Field(default=64, gt=0)

    @model_validator(mode="after")
    def deterministic_sampling(self) -> Self:
        if self.temperature != 0:
            raise ValueError("The deterministic fake provider requires temperature=0")
        return self


class ExperimentConfig(FrozenModel):
    experiment_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    seed: int = Field(ge=0, le=2**32 - 1)
    environment: EnvironmentConfig
    task: str = Field(min_length=1)
    agent: AgentConfig
    model: ModelConfig
    max_steps: int = Field(gt=0)

    @model_validator(mode="after")
    def supported_task(self) -> Self:
        expected = "repair_pump" if self.environment.name == "fake" else "collect_wood"
        if self.task != expected:
            raise ValueError(f"{self.environment.name} currently supports task {expected}")
        if self.environment.name == "alem" and self.model.policy != "first_available":
            raise ValueError(
                "Alem smoke requires the environment-independent first_available policy"
            )
        return self


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
