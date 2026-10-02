"""Predeclared, teacher-free trials; every seed is run once without repair or retries."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from mosaic.core.config import ExperimentConfig, ModelConfig, ValidationConfig
from mosaic.core.provenance import configuration_sha256
from mosaic.environments.base import EnvironmentAdapter
from mosaic.experiments.runner import ExperimentRunner, RunArtifacts
from mosaic.providers.base import ModelProvider
from mosaic.skills.models import CandidateSkill, ValidationReport, ValidationTrial


class SkillValidator:
    def __init__(
        self,
        environment_factory: Callable[[ExperimentConfig], EnvironmentAdapter],
        weak_provider_factory: Callable[[ModelConfig], ModelProvider],
    ) -> None:
        self.environment_factory = environment_factory
        self.weak_provider_factory = weak_provider_factory
        # Audit artifacts only; they are never provided to an agent or a later trial.
        self.completed_runs: tuple[RunArtifacts, ...] = ()

    async def validate(
        self,
        candidate: CandidateSkill,
        base_config: ExperimentConfig,
        settings: ValidationConfig,
        output_dir: Path,
    ) -> ValidationReport:
        self.completed_runs = ()
        # Refuse to overwrite earlier evidence or silently rerun an existing trial namespace.
        output_dir.mkdir(parents=True, exist_ok=False)
        trials: list[ValidationTrial] = []
        for seed in settings.seeds:
            data = {
                **base_config.model_dump(mode="json"),
                "teacher": None,
                "routing": None,
                "seed": seed,
                "max_steps": settings.max_steps,
                "skills": {"mode": "validation", "candidate_id": candidate.skill_id},
                "experiment_id": f"{base_config.experiment_id}-validation-{seed}",
            }
            config = ExperimentConfig.model_validate_json(json.dumps(data))
            environment = self.environment_factory(config)
            weak_provider = self.weak_provider_factory(config.model)
            result = await ExperimentRunner(environment, weak_provider, candidate=candidate).run(
                config, output_dir / str(seed)
            )
            self.completed_runs = (*self.completed_runs, result)
            trials.append(
                ValidationTrial(
                    seed=seed,
                    config_sha256=configuration_sha256(config),
                    trace_sha256=hashlib.sha256(result.trace_path.read_bytes()).hexdigest(),
                    provider=config.model.provider,
                    model=config.model.model,
                    status=result.summary.status,
                    reason=result.summary.reason,
                    skill_completed=result.summary.skill_completed,
                    budget=result.summary.budget,
                )
            )
        return ValidationReport(
            skill_id=candidate.skill_id,
            version=candidate.version,
            content_sha256=candidate.content_sha256,
            trials=tuple(trials),
        )
