import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from test_experiments import semantic

from mosaic.core.config import (
    ExperimentConfig,
    LearningConfig,
    ModelConfig,
    load_learning_config,
)
from mosaic.core.events import (
    ActionDispatched,
    ExperimentStarted,
    ModelCalled,
    ModelCallFailed,
    ModelResponded,
    RunErrored,
    SkillCandidateCreated,
    SkillCompilationStarted,
    SkillExecutionFailed,
    SkillExecutionFinished,
    SkillExecutionStarted,
    SkillLearningFinished,
    SkillLearningStarted,
    SkillPromoted,
    SkillRejected,
    SkillRetrievalPerformed,
    SkillValidated,
    SkillValidationStarted,
    read_events,
)
from mosaic.core.models import ModelRequest, ModelResponse
from mosaic.environments.base import EnvironmentAdapter
from mosaic.experiments.learn import environment_factory, provider_factory
from mosaic.experiments.learning import LearningArtifacts, LearningRunner, LearningSummary
from mosaic.experiments.runner import ExperimentRunner
from mosaic.providers.base import ModelProvider
from mosaic.providers.fake import FakeModelProvider
from mosaic.skills.models import CandidateSkill, ValidationReport
from mosaic.skills.playbook import PlayBook

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def learning_config() -> LearningConfig:
    return load_learning_config(PROJECT / "configs" / "skill-pilot.yaml")


def changed(config: LearningConfig, **changes: Any) -> LearningConfig:
    return LearningConfig.model_validate_json(
        json.dumps({**config.model_dump(mode="json"), **changes})
    )


def run_learning(config: LearningConfig, output: Path) -> LearningArtifacts:
    return asyncio.run(
        LearningRunner(environment_factory, provider_factory, provider_factory).run(config, output)
    )


def test_full_lifecycle_reconciles_costs_and_injected_context(
    learning_config: LearningConfig, tmp_path: Path
) -> None:
    output = tmp_path / "learn"
    result = run_learning(learning_config, output)
    summary = result.summary
    assert LearningSummary.model_validate_json(result.summary_path.read_text()) == summary
    assert summary.status == "succeeded" and summary.promoted and summary.budget_complete
    assert summary.reason == "validated_skill_reused"
    assert summary.compilation_calls == 1 and summary.compilation_model_calls == 0
    assert summary.compilation_ms >= 0
    assert summary.acquisition is not None and summary.validation is not None
    assert summary.acquisition.budget.weak.model_calls == 2
    assert summary.acquisition.budget.teacher.model_calls == 16
    assert summary.acquisition.budget.environment_actions == 18
    assert summary.validation.passed and len(summary.validation.trials) == 1
    assert summary.validation_budget.weak.model_calls == 16
    assert summary.reuse_budget.weak.model_calls == 16
    assert summary.budget.weak.model_calls == 34
    assert summary.budget.teacher.model_calls == 16
    assert summary.budget.model_calls == summary.budget.environment_actions == 50
    assert (
        summary.validation_budget.teacher.model_calls
        == summary.reuse_budget.teacher.model_calls
        == 0
    )

    events = read_events(result.trace_path)
    assert [event.sequence for event in events] == list(range(len(events)))
    assert [type(event.payload) for event in events] == [
        SkillLearningStarted,
        SkillCompilationStarted,
        SkillCandidateCreated,
        SkillValidationStarted,
        SkillValidated,
        SkillPromoted,
        SkillLearningFinished,
    ]
    candidate = CandidateSkill.model_validate_json((output / "candidate.json").read_text())
    report = ValidationReport.model_validate_json((output / "validation.json").read_text())
    entries = PlayBook(result.playbook_path).entries()
    assert (
        len(entries) == 1 and entries[0].candidate == candidate and entries[0].validation == report
    )
    assert candidate.provenance.source_agent == "agent_0"
    assert len(candidate.procedure) == 16
    assert summary.skill_id == candidate.skill_id
    assert summary.playbook_sha256 == PlayBook(result.playbook_path).fingerprint()

    phase_paths = (
        output / "acquisition" / "events.jsonl",
        output / "validation" / "19" / "events.jsonl",
        output / "reuse" / "20" / "events.jsonl",
    )
    phase_events = tuple(event for path in phase_paths for event in read_events(path))
    calls = [event.payload for event in phase_events if isinstance(event.payload, ModelCalled)]
    responses = [
        event.payload for event in phase_events if isinstance(event.payload, ModelResponded)
    ]
    assert summary.budget.environment_actions == sum(
        isinstance(event.payload, ActionDispatched) for event in phase_events
    )
    for role, budget in (("weak", summary.budget.weak), ("teacher", summary.budget.teacher)):
        role_calls = [call for call in calls if call.model_role == role]
        role_responses = [response for response in responses if response.model_role == role]
        assert budget.model_calls == len(role_calls) == len(role_responses)
        # Synthetic usage measures the entire serialized request, including each injected procedure.
        assert budget.input_tokens == sum(
            len(call.request.model_dump_json().split()) for call in role_calls
        )
        assert budget.input_tokens == sum(
            response.response.usage.input_tokens or 0 for response in role_responses
        )
        assert budget.output_tokens == sum(
            response.response.usage.output_tokens or 0 for response in role_responses
        )
        assert budget.synthetic_usage_calls == budget.model_calls
        assert budget.wall_clock_ms == pytest.approx(
            sum(response.latency_ms for response in role_responses)
        )
    prompt_bytes = sum(
        len(call.request.skill.model_dump_json().encode("utf-8"))
        for call in calls
        if call.request.skill is not None
    )
    assert summary.budget.skill_prompt_bytes == prompt_bytes > 0
    assert summary.budget.input_tokens == sum(
        response.response.usage.input_tokens or 0 for response in responses
    )
    assert summary.budget.output_tokens == sum(
        response.response.usage.output_tokens or 0 for response in responses
    )
    assert summary.budget.wall_clock_ms == pytest.approx(
        summary.acquisition_budget.wall_clock_ms
        + summary.validation_budget.wall_clock_ms
        + summary.reuse_budget.wall_clock_ms
    )
    assert summary.pipeline_wall_clock_ms >= summary.budget.wall_clock_ms
    assert summary.acquisition_budget.skill_prompt_bytes == 0
    assert summary.validation_budget.skill_prompt_bytes == summary.reuse_budget.skill_prompt_bytes
    assert summary.budget.skill_retrievals == 1
    reuse_events = read_events(phase_paths[2])
    start = next(
        event.payload for event in reuse_events if isinstance(event.payload, ExperimentStarted)
    )
    assert start.config.agent.agent_id == "agent_1"
    assert start.config.teacher is None and start.config.routing is None
    assert all(event.agent_id == "agent_1" for event in reuse_events)
    skill_kinds = [
        type(event.payload)
        for event in reuse_events
        if isinstance(
            event.payload, (SkillRetrievalPerformed, SkillExecutionStarted, SkillExecutionFinished)
        )
    ]
    assert skill_kinds == [SkillRetrievalPerformed, SkillExecutionStarted, SkillExecutionFinished]
    retrieval = next(
        event.payload
        for event in reuse_events
        if isinstance(event.payload, SkillRetrievalPerformed)
    )
    assert (
        retrieval.skill_id == candidate.skill_id
        and retrieval.playbook_sha256 == summary.playbook_sha256
    )


def test_every_predeclared_validation_seed_is_required(
    learning_config: LearningConfig, tmp_path: Path
) -> None:
    config = changed(
        learning_config,
        validation={**learning_config.validation.model_dump(mode="json"), "seeds": [42, 43]},
    )
    result = run_learning(config, tmp_path / "learn")
    assert result.summary.status == "failed" and result.summary.reason == "weak_validation_failed"
    report = result.summary.validation
    assert report is not None and not report.passed
    assert [trial.seed for trial in report.trials] == [42, 43]
    assert report.trials[0].status == "succeeded" and report.trials[0].skill_completed
    assert report.trials[1].status == "failed" and not report.trials[1].skill_completed
    assert report.trials[1].reason == "skill_initiation_not_matched"
    assert result.summary.validation_budget.weak.model_calls == 16
    assert result.summary.validation_budget.teacher.model_calls == 0
    assert not result.summary.promoted and result.summary.reuse == ()
    assert not result.playbook_path.exists()
    assert not any(
        isinstance(event.payload, SkillPromoted) for event in read_events(result.trace_path)
    )


def test_source_weak_success_cannot_create_teacher_skill(
    learning_config: LearningConfig, tmp_path: Path
) -> None:
    acquisition = learning_config.acquisition.model_dump(mode="json")
    acquisition["model"] = {
        **learning_config.acquisition.model.model_dump(),
        "policy": "repair_pump",
    }
    assert learning_config.acquisition.routing is not None
    acquisition["routing"] = {
        **learning_config.acquisition.routing.model_dump(),
        "attempt_steps": 20,
        "recovery_steps": 20,
    }
    result = run_learning(changed(learning_config, acquisition=acquisition), tmp_path / "learn")
    assert result.summary.acquisition is not None
    assert result.summary.acquisition.status == "succeeded"
    assert result.summary.budget.weak.model_calls == 16
    assert result.summary.budget.teacher.model_calls == result.summary.compilation_calls == 0
    assert not result.summary.promoted and result.summary.validation is None
    assert not result.playbook_path.exists()
    assert not (tmp_path / "learn" / "candidate.json").exists()
    assert [type(event.payload) for event in read_events(result.trace_path)] == [
        SkillLearningStarted,
        SkillRejected,
        SkillLearningFinished,
    ]


def test_unsuccessful_teacher_never_compiles_or_promotes(
    learning_config: LearningConfig, tmp_path: Path
) -> None:
    acquisition = learning_config.acquisition.model_dump(mode="json")
    assert learning_config.acquisition.teacher is not None
    acquisition["teacher"] = {
        **learning_config.acquisition.teacher.model_dump(mode="json"),
        "model": {
            **learning_config.acquisition.teacher.model.model_dump(),
            "policy": "first_available",
        },
    }
    result = run_learning(changed(learning_config, acquisition=acquisition), tmp_path / "learn")
    assert result.summary.status == "failed" and result.summary.acquisition is not None
    assert result.summary.acquisition.status == "failed"
    assert result.summary.budget.weak.model_calls == 2
    assert result.summary.budget.teacher.model_calls == 30
    assert result.summary.compilation_calls == 0 and result.summary.validation is None
    assert not result.summary.promoted and not result.playbook_path.exists()


def test_ignored_candidate_fails_validation_after_one_divergent_action(
    learning_config: LearningConfig, tmp_path: Path
) -> None:
    def ignoring_provider(config: ModelConfig) -> FakeModelProvider:
        return FakeModelProvider("first_available")

    result = asyncio.run(
        LearningRunner(environment_factory, ignoring_provider, provider_factory).run(
            learning_config, tmp_path / "learn"
        )
    )
    report = result.summary.validation
    assert report is not None and not report.passed and len(report.trials) == 1
    assert report.trials[0].reason == "skill_execution_failed"
    assert report.trials[0].budget.weak.model_calls == 1
    assert report.trials[0].budget.skill_execution_failures == 1
    assert not report.trials[0].skill_completed and not result.summary.promoted
    assert not result.playbook_path.exists()
    failures = [
        event.payload
        for event in read_events(tmp_path / "learn" / "validation" / "19" / "events.jsonl")
        if isinstance(event.payload, SkillExecutionFailed)
    ]
    assert len(failures) == 1 and failures[0].completed_actions == 0
    assert failures[0].reason == "weak_action_rejected_or_diverged"


class FailingTeacher:
    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise RuntimeError("Teacher transport unavailable")


def test_failed_model_call_usage_survives_pipeline_aggregation(
    learning_config: LearningConfig, tmp_path: Path
) -> None:
    def failing_provider(config: ModelConfig) -> FailingTeacher:
        return FailingTeacher()

    result = asyncio.run(
        LearningRunner(environment_factory, provider_factory, failing_provider).run(
            learning_config, tmp_path / "learn"
        )
    )
    assert result.summary.acquisition is not None and result.summary.acquisition.status == "error"
    assert result.summary.status == "error" and result.summary.reason.startswith("acquisition:")
    budget = result.summary.budget
    assert (
        budget.model_calls == 3 and budget.weak.model_calls == 2 and budget.teacher.model_calls == 1
    )
    assert budget.weak.input_tokens is not None and budget.weak.output_tokens is not None
    assert budget.teacher.input_tokens is budget.teacher.output_tokens is None
    assert budget.input_tokens is budget.output_tokens is None
    assert budget.known_input_tokens == budget.weak.known_input_tokens > 0
    assert budget.unknown_input_calls == budget.unknown_output_calls == 1
    assert budget.teacher.unknown_input_calls == budget.teacher.unknown_output_calls == 1
    assert budget.synthetic_usage_calls == 2 and budget.environment_actions == 2
    failure = next(
        event.payload
        for event in read_events(tmp_path / "learn" / "acquisition" / "events.jsonl")
        if isinstance(event.payload, ModelCallFailed)
    )
    assert budget.teacher.wall_clock_ms == failure.latency_ms
    assert not result.summary.promoted and result.summary.compilation_calls == 0


def test_partial_validation_setup_error_retains_completed_trial_cost(
    learning_config: LearningConfig, tmp_path: Path
) -> None:
    config = changed(
        learning_config,
        validation={**learning_config.validation.model_dump(mode="json"), "seeds": [42, 43]},
    )

    def failing_environment(config: ExperimentConfig) -> EnvironmentAdapter:
        if config.skills is not None and config.skills.mode == "validation" and config.seed == 43:
            raise RuntimeError("Second validation environment failed to initialize")
        return environment_factory(config)

    result = asyncio.run(
        LearningRunner(failing_environment, provider_factory, provider_factory).run(
            config, tmp_path / "learn"
        )
    )
    assert result.summary.status == "error" and result.summary.reason.startswith("validation:")
    assert not result.summary.budget_complete and result.summary.validation is None
    assert result.summary.validation_budget.weak.model_calls == 16
    assert result.summary.validation_budget.environment_actions == 16
    assert result.summary.budget.weak.model_calls == 18
    assert result.summary.budget.teacher.model_calls == 16
    assert result.summary.budget.environment_actions == 34
    assert (tmp_path / "learn" / "validation" / "42" / "summary.json").exists()
    assert not result.playbook_path.exists() and not result.summary.promoted


def test_existing_output_is_preserved_before_starting_pipeline(
    learning_config: LearningConfig, tmp_path: Path
) -> None:
    output = tmp_path / "learn"
    output.mkdir()
    sentinel = output / "events.jsonl"
    sentinel.write_bytes(b"Earlier evidence must remain intact\n")
    with pytest.raises(FileExistsError):
        run_learning(learning_config, output)
    assert sentinel.read_bytes() == b"Earlier evidence must remain intact\n"
    assert sorted(path.name for path in output.iterdir()) == ["events.jsonl"]


@pytest.mark.parametrize(
    "case,exit_code", [("success", 0), ("failed_validation", 1), ("invalid", 2)]
)
def test_learning_cli_reports_distinct_outcomes(
    learning_config: LearningConfig, tmp_path: Path, case: str, exit_code: int
) -> None:
    data = learning_config.model_dump(mode="json")
    if case == "failed_validation":
        data["validation"]["seeds"] = [42, 43]
    elif case == "invalid":
        data["validation"]["seeds"] = []
    config_path = tmp_path / "config.yaml"
    # JSON is a YAML subset and avoids coupling this CLI test to serializer formatting.
    config_path.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "learn"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mosaic.experiments.learn",
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
        ],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == exit_code, completed.stdout + completed.stderr
    if exit_code == 2:
        assert "Cannot start skill experiment" in completed.stderr and not output.exists()
    else:
        summary = LearningSummary.model_validate_json((output / "summary.json").read_text())
        assert summary.status == ("succeeded" if exit_code == 0 else "failed")
        assert "All phases: weak calls" in completed.stdout


def test_standalone_persisted_reuse_requires_matching_context_and_no_teacher(
    learning_config: LearningConfig, tmp_path: Path
) -> None:
    learned = run_learning(learning_config, tmp_path / "learn")
    source = learning_config.acquisition.model_dump(mode="json")
    base = {
        **source,
        "teacher": None,
        "routing": None,
        "agent": {"agent_id": "new_agent"},
        "skills": {"mode": "reuse", "playbook_path": str(learned.playbook_path.resolve())},
    }
    for label, seed, use_book, expected_status in (
        ("matching", 42, True, "succeeded"),
        ("different_context", 43, True, "failed"),
        ("baseline", 42, False, "failed"),
    ):
        config = ExperimentConfig.model_validate_json(
            json.dumps(
                {
                    **base,
                    "experiment_id": label,
                    "seed": seed,
                    "skills": base["skills"] if use_book else None,
                }
            )
        )
        result = asyncio.run(
            ExperimentRunner(
                environment_factory(config),
                provider_factory(config.model),
                playbook=PlayBook(learned.playbook_path) if use_book else None,
            ).run(config, tmp_path / label)
        )
        assert result.summary.status == expected_status
        assert result.summary.budget.teacher.model_calls == 0 and result.summary.escalations == 0
        events = read_events(result.trace_path)
        assert all(event.agent_id == "new_agent" for event in events)
        calls = [event.payload for event in events if isinstance(event.payload, ModelCalled)]
        if expected_status == "succeeded":
            assert result.summary.skill_completed and result.summary.budget.weak.model_calls == 16
            assert all(call.request.skill is not None for call in calls)
        else:
            assert not result.summary.skill_completed and result.summary.skill_id is None
            assert result.summary.budget.weak.model_calls == config.max_steps
            assert result.summary.budget.skill_prompt_bytes == 0
            assert all(call.request.skill is None for call in calls)
            if use_book:
                retrieval = next(
                    event.payload
                    for event in events
                    if isinstance(event.payload, SkillRetrievalPerformed)
                )
                assert retrieval.skill_id is None


def test_fixed_persisted_playbook_has_deterministic_reuse_trace(
    learning_config: LearningConfig, tmp_path: Path
) -> None:
    learned = run_learning(learning_config, tmp_path / "learn")
    config = ExperimentConfig.model_validate_json(
        json.dumps(
            {
                **learning_config.acquisition.model_dump(mode="json"),
                "teacher": None,
                "routing": None,
                "agent": {"agent_id": "agent_1"},
                "skills": {
                    "mode": "reuse",
                    "playbook_path": str(learned.playbook_path.resolve()),
                },
            }
        )
    )
    traces = []
    for label in ("first", "second"):
        result = asyncio.run(
            ExperimentRunner(
                environment_factory(config),
                provider_factory(config.model),
                playbook=PlayBook(learned.playbook_path),
            ).run(config, tmp_path / label)
        )
        assert result.summary.status == "succeeded" and result.summary.skill_completed
        traces.append(
            [semantic(event.model_dump(mode="json")) for event in read_events(result.trace_path)]
        )
    assert traces[0] == traces[1]


def test_corrupt_persisted_playbook_counts_failed_retrieval_without_fallback(
    learning_config: LearningConfig, tmp_path: Path
) -> None:
    learned = run_learning(learning_config, tmp_path / "learn")
    corrupted = b'{"schema_version":1,"entries":['
    learned.playbook_path.write_bytes(corrupted)
    config = ExperimentConfig.model_validate_json(
        json.dumps(
            {
                **learning_config.acquisition.model_dump(mode="json"),
                "teacher": None,
                "routing": None,
                "agent": {"agent_id": "new_agent"},
                "skills": {
                    "mode": "reuse",
                    "playbook_path": str(learned.playbook_path.resolve()),
                },
            }
        )
    )
    result = asyncio.run(
        ExperimentRunner(
            environment_factory(config),
            provider_factory(config.model),
            playbook=PlayBook(learned.playbook_path),
        ).run(config, tmp_path / "corrupted-reuse")
    )
    assert result.summary.status == "error" and result.summary.reason.startswith("skill_retrieval:")
    assert result.summary.budget.skill_retrievals == 1
    assert result.summary.budget.skill_retrieval_ms > 0
    assert result.summary.budget.model_calls == result.summary.budget.environment_actions == 0
    assert result.summary.budget.skill_prompt_bytes == 0
    assert not result.summary.skill_completed and result.summary.skill_id is None
    events = read_events(result.trace_path)
    errors = [event.payload for event in events if isinstance(event.payload, RunErrored)]
    assert len(errors) == 1 and errors[0].stage == "skill_retrieval"
    assert not any(isinstance(event.payload, ModelCalled) for event in events)
    assert learned.playbook_path.read_bytes() == corrupted


class FailingWeak:
    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise RuntimeError("Weak provider transport unavailable")


@pytest.mark.parametrize("phase", ["validation", "reuse"])
def test_weak_transport_failure_retains_costs_and_continues_declared_episodes(
    learning_config: LearningConfig, tmp_path: Path, phase: str
) -> None:
    # Seeds 42 and 19 have the same visible initiation in this fixture; the second
    # declared episode therefore demonstrates continued weak execution after an error.
    config = changed(
        learning_config,
        validation={
            **learning_config.validation.model_dump(mode="json"),
            "seeds": [42, 19] if phase == "validation" else [42],
        },
        reuse={**learning_config.reuse.model_dump(mode="json"), "seeds": [42, 19]},
    )
    provider_instances = 0
    failing_instance = 2 if phase == "validation" else 3

    def phase_provider(settings: ModelConfig) -> ModelProvider:
        nonlocal provider_instances
        provider_instances += 1
        if provider_instances == failing_instance:
            return FailingWeak()
        return provider_factory(settings)

    output = tmp_path / "learn"
    result = asyncio.run(
        LearningRunner(environment_factory, phase_provider, provider_factory).run(config, output)
    )
    summary = result.summary
    assert summary.status == "error" and summary.budget_complete
    expected_reason = "weak_validation_errored" if phase == "validation" else "skill_reuse_errored"
    assert summary.reason == expected_reason
    assert summary.acquisition is not None and summary.acquisition.status == "succeeded"
    assert summary.acquisition_budget.weak.model_calls == 2
    assert summary.acquisition_budget.teacher.model_calls == 16
    assert summary.acquisition_budget.environment_actions == 18
    assert summary.validation is not None
    phase_paths: tuple[Path, ...]
    if phase == "validation":
        trials = summary.validation.trials
        assert [trial.seed for trial in trials] == [42, 19]
        assert [trial.status for trial in trials] == ["error", "succeeded"]
        assert not trials[0].skill_completed and trials[1].skill_completed
        assert summary.validation_budget.weak.model_calls == 17
        assert summary.validation_budget.environment_actions == 16
        assert provider_instances == 3
        assert not summary.promoted and summary.reuse == ()
        assert summary.budget.weak.model_calls == 19
        assert summary.budget.environment_actions == 34
        phase_paths = (
            output / "acquisition" / "events.jsonl",
            output / "validation" / "42" / "events.jsonl",
            output / "validation" / "19" / "events.jsonl",
        )
    else:
        assert summary.validation.passed
        assert summary.validation_budget.weak.model_calls == 16
        assert [episode.status for episode in summary.reuse] == ["error", "succeeded"]
        assert not summary.reuse[0].skill_completed and summary.reuse[1].skill_completed
        assert summary.reuse_budget.weak.model_calls == 17
        assert summary.reuse_budget.environment_actions == 16
        assert provider_instances == 4 and summary.promoted
        assert summary.budget.weak.model_calls == 35
        assert summary.budget.environment_actions == 50
        phase_paths = (
            output / "acquisition" / "events.jsonl",
            output / "validation" / "42" / "events.jsonl",
            output / "reuse" / "42" / "events.jsonl",
            output / "reuse" / "19" / "events.jsonl",
        )
    assert summary.budget.teacher.model_calls == 16
    assert summary.budget.input_tokens is summary.budget.output_tokens is None
    assert summary.budget.weak.input_tokens is summary.budget.weak.output_tokens is None
    assert summary.budget.teacher.input_tokens is not None
    assert summary.budget.unknown_input_calls == summary.budget.unknown_output_calls == 1
    assert summary.budget.weak.unknown_input_calls == summary.budget.weak.unknown_output_calls == 1
    assert summary.budget.synthetic_usage_calls == summary.budget.model_calls - 1
    events = tuple(event for path in phase_paths for event in read_events(path))
    responses = [event.payload for event in events if isinstance(event.payload, ModelResponded)]
    failures = [event.payload for event in events if isinstance(event.payload, ModelCallFailed)]
    assert len(failures) == 1 and failures[0].model_role == "weak"
    assert summary.budget.known_input_tokens == sum(
        response.response.usage.input_tokens or 0 for response in responses
    )
    assert summary.budget.known_output_tokens == sum(
        response.response.usage.output_tokens or 0 for response in responses
    )
    assert summary.budget.weak.wall_clock_ms == pytest.approx(
        sum(response.latency_ms for response in responses if response.model_role == "weak")
        + failures[0].latency_ms
    )
