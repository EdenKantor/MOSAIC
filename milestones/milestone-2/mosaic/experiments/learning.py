"""Explicit acquisition -> bounded compilation -> weak validation -> promotion -> reuse."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from mosaic.core.budget import BudgetLedger, BudgetTotals, RoleBudgetTotals
from mosaic.core.config import ExperimentConfig, LearningConfig, ModelConfig
from mosaic.core.events import (
    EventRecorder,
    RunErrored,
    RunSummary,
    SkillCandidateCreated,
    SkillCompilationStarted,
    SkillLearningFinished,
    SkillLearningStarted,
    SkillPromoted,
    SkillRejected,
    SkillValidated,
    SkillValidationStarted,
)
from mosaic.core.models import FrozenModel
from mosaic.core.provenance import configuration_sha256
from mosaic.core.teacher import TeacherCapability
from mosaic.environments.base import EnvironmentAdapter
from mosaic.experiments.runner import ExperimentRunner, RunArtifacts
from mosaic.providers.base import ModelProvider
from mosaic.skills.compiler import compile_candidate, extract_teacher_trajectory
from mosaic.skills.models import CandidateSkill, ValidationReport, VerifiedSkill
from mosaic.skills.playbook import PlayBook
from mosaic.skills.validation import SkillValidator


def _merge_dimension(
    values: tuple[RoleBudgetTotals, ...], dimension: str, unknown_field: str
) -> tuple[int | None, int, int]:
    known = sum(getattr(value, f"known_{dimension}") for value in values)
    unknown = sum(getattr(value, unknown_field) for value in values)
    return None if unknown else known, known, unknown


def _merge_role(values: tuple[RoleBudgetTotals, ...]) -> RoleBudgetTotals:
    known_in = sum(value.known_input_tokens for value in values)
    known_out = sum(value.known_output_tokens for value in values)
    unknown_in = sum(value.unknown_input_calls for value in values)
    unknown_out = sum(value.unknown_output_calls for value in values)
    reasoning, known_reasoning, unknown_reasoning = _merge_dimension(
        values, "reasoning_tokens", "unknown_reasoning_calls"
    )
    cached, known_cached, unknown_cached = _merge_dimension(
        values, "cached_input_tokens", "unknown_cached_input_calls"
    )
    provider_total, known_total, unknown_total = _merge_dimension(
        values, "provider_total_tokens", "unknown_provider_total_calls"
    )
    return RoleBudgetTotals(
        model_calls=sum(value.model_calls for value in values),
        input_tokens=None if unknown_in else known_in,
        output_tokens=None if unknown_out else known_out,
        known_input_tokens=known_in,
        known_output_tokens=known_out,
        unknown_input_calls=unknown_in,
        unknown_output_calls=unknown_out,
        synthetic_usage_calls=sum(value.synthetic_usage_calls for value in values),
        wall_clock_ms=sum(value.wall_clock_ms for value in values),
        reasoning_tokens=reasoning,
        cached_input_tokens=cached,
        provider_total_tokens=provider_total,
        known_reasoning_tokens=known_reasoning,
        known_cached_input_tokens=known_cached,
        known_provider_total_tokens=known_total,
        unknown_reasoning_calls=unknown_reasoning,
        unknown_cached_input_calls=unknown_cached,
        unknown_provider_total_calls=unknown_total,
    )


def merge_budgets(values: tuple[BudgetTotals, ...]) -> BudgetTotals:
    """Sum raw phase dimensions; unknown call usage remains unknown."""
    return BudgetTotals(
        **_merge_role(values).model_dump(),
        weak=_merge_role(tuple(value.weak for value in values)),
        teacher=_merge_role(tuple(value.teacher for value in values)),
        environment_actions=sum(value.environment_actions for value in values),
        skill_retrievals=sum(value.skill_retrievals for value in values),
        skill_retrieval_ms=sum(value.skill_retrieval_ms for value in values),
        skill_prompt_bytes=sum(value.skill_prompt_bytes for value in values),
        skill_execution_failures=sum(value.skill_execution_failures for value in values),
    )


class LearningSummary(FrozenModel):
    experiment_id: str
    status: Literal["succeeded", "failed", "error"]
    reason: str
    config_sha256: str
    skill_id: str | None
    promoted: bool
    acquisition: RunSummary | None
    validation: ValidationReport | None
    reuse: tuple[RunSummary, ...]
    acquisition_budget: BudgetTotals
    validation_budget: BudgetTotals
    reuse_budget: BudgetTotals
    budget: BudgetTotals
    budget_complete: bool
    compilation_calls: int
    compilation_model_calls: Literal[0] = 0
    compilation_ms: float
    pipeline_wall_clock_ms: float
    playbook_sha256: str | None


@dataclass(frozen=True)
class LearningArtifacts:
    trace_path: Path
    summary_path: Path
    playbook_path: Path
    summary: LearningSummary


class LearningRunner:
    def __init__(
        self,
        environment_factory: Callable[[ExperimentConfig], EnvironmentAdapter],
        weak_provider_factory: Callable[[ModelConfig], ModelProvider],
        teacher_provider_factory: Callable[[ModelConfig], ModelProvider],
    ) -> None:
        self.environment_factory = environment_factory
        self.weak_provider_factory = weak_provider_factory
        self.teacher_provider_factory = teacher_provider_factory

    async def run(self, config: LearningConfig, output_dir: Path) -> LearningArtifacts:
        output_dir.mkdir(parents=True, exist_ok=False)
        trace, summary_path = output_dir / "events.jsonl", output_dir / "summary.json"
        book = PlayBook(output_dir / "playbook.json")
        recorder = EventRecorder(
            trace, config.experiment_id, str(uuid4()), config.acquisition.agent.agent_id
        )
        started = time.perf_counter()
        acquisition: RunArtifacts | None = None
        candidate: CandidateSkill | None = None
        report: ValidationReport | None = None
        reuses: list[RunArtifacts] = []
        validator = SkillValidator(self.environment_factory, self.weak_provider_factory)
        status: Literal["succeeded", "failed", "error"] = "failed"
        reason, stage = "source_not_verified_teacher_success", "acquisition"
        compilation_calls, compilation_ms, promoted, budget_complete = 0, 0.0, False, True
        try:
            recorder.record(SkillLearningStarted(config=config))
            source_config = config.acquisition
            assert source_config.teacher is not None
            acquisition = await ExperimentRunner(
                self.environment_factory(source_config),
                self.weak_provider_factory(source_config.model),
                TeacherCapability(
                    self.teacher_provider_factory(source_config.teacher.model),
                    source_config.teacher.access,
                ),
            ).run(source_config, output_dir / "acquisition")
            if (
                acquisition.summary.status != "succeeded"
                or not acquisition.summary.budget.teacher.model_calls
            ):
                if acquisition.summary.status == "error":
                    status, reason = "error", f"acquisition: {acquisition.summary.reason}"
                recorder.record(SkillRejected(skill_id=None, reason=reason))
            else:
                stage = "compilation"
                recorder.record(
                    SkillCompilationStarted(source_episode=acquisition.summary.episode_id)
                )
                compilation_calls = 1
                compile_started = time.perf_counter()
                try:
                    candidate = compile_candidate(
                        extract_teacher_trajectory(acquisition), config.compilation
                    )
                finally:
                    compilation_ms = (time.perf_counter() - compile_started) * 1000
                (output_dir / "candidate.json").write_text(
                    candidate.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
                recorder.record(
                    SkillCandidateCreated(candidate=candidate, compilation_ms=compilation_ms)
                )
                stage = "validation"
                recorder.record(
                    SkillValidationStarted(
                        skill_id=candidate.skill_id, seeds=config.validation.seeds
                    )
                )
                report = await validator.validate(
                    candidate, source_config, config.validation, output_dir / "validation"
                )
                (output_dir / "validation.json").write_text(
                    report.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
                if not report.passed:
                    if any(trial.status == "error" for trial in report.trials):
                        status, reason = "error", "weak_validation_errored"
                    else:
                        reason = "weak_validation_failed"
                    recorder.record(
                        SkillRejected(skill_id=candidate.skill_id, reason=reason, report=report)
                    )
                else:
                    recorder.record(SkillValidated(report=report))
                    stage = "promotion"
                    inserted = book.promote(candidate, report)
                    promoted = True
                    book_hash = book.fingerprint()
                    assert book_hash is not None
                    recorder.record(
                        SkillPromoted(
                            skill=VerifiedSkill(candidate=candidate, validation=report),
                            playbook_sha256=book_hash,
                            inserted=inserted,
                        )
                    )
                    stage = "reuse"
                    for seed in config.reuse.seeds:
                        reuse_config = ExperimentConfig.model_validate_json(
                            json.dumps(
                                {
                                    **source_config.model_dump(mode="json"),
                                    "experiment_id": f"{config.experiment_id}-reuse-{seed}",
                                    "seed": seed,
                                    "teacher": None,
                                    "routing": None,
                                    "agent": {"agent_id": config.reuse.agent_id},
                                    "max_steps": config.reuse.max_steps,
                                    "skills": {
                                        "mode": "reuse",
                                        "playbook_path": str(book.path.resolve()),
                                    },
                                }
                            )
                        )
                        reuses.append(
                            await ExperimentRunner(
                                self.environment_factory(reuse_config),
                                self.weak_provider_factory(reuse_config.model),
                                playbook=PlayBook(book.path),
                            ).run(reuse_config, output_dir / "reuse" / str(seed))
                        )
                    if all(
                        run.summary.status == "succeeded" and run.summary.skill_completed
                        for run in reuses
                    ):
                        status, reason = "succeeded", "validated_skill_reused"
                    else:
                        if any(run.summary.status == "error" for run in reuses):
                            status, reason = "error", "skill_reuse_errored"
                        else:
                            reason = "skill_reuse_failed"
        except Exception as error:
            status, reason = "error", f"{stage}: {type(error).__name__}: {error}"
            if stage == "compilation" and isinstance(error, ValueError):
                status, reason = "failed", f"compilation_rejected: {error}"
                recorder.record(SkillRejected(skill_id=None, reason=reason))
            else:
                budget_complete = False
                recorder.record(
                    RunErrored(stage=stage, error_type=type(error).__name__, message=str(error))
                )
        finally:
            try:
                acquisition_budget = (
                    acquisition.summary.budget if acquisition else BudgetLedger().totals(0)
                )
                validation_budget = merge_budgets(
                    tuple(run.summary.budget for run in validator.completed_runs)
                )
                reuse_budget = merge_budgets(tuple(run.summary.budget for run in reuses))
                budget = merge_budgets((acquisition_budget, validation_budget, reuse_budget))
                summary = LearningSummary(
                    experiment_id=config.experiment_id,
                    status=status,
                    reason=reason,
                    config_sha256=configuration_sha256(config),
                    skill_id=candidate.skill_id if candidate else None,
                    promoted=promoted,
                    acquisition=acquisition.summary if acquisition else None,
                    validation=report,
                    reuse=tuple(run.summary for run in reuses),
                    acquisition_budget=acquisition_budget,
                    validation_budget=validation_budget,
                    reuse_budget=reuse_budget,
                    budget=budget,
                    budget_complete=budget_complete,
                    compilation_calls=compilation_calls,
                    compilation_ms=compilation_ms,
                    pipeline_wall_clock_ms=(time.perf_counter() - started) * 1000,
                    playbook_sha256=book.fingerprint(),
                )
                recorder.record(
                    SkillLearningFinished(
                        status=status,
                        reason=reason,
                        skill_id=summary.skill_id,
                        budget=budget,
                        compilation_calls=compilation_calls,
                        compilation_ms=compilation_ms,
                    )
                )
                summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
            finally:
                recorder.close()
        return LearningArtifacts(trace, summary_path, book.path, summary)
