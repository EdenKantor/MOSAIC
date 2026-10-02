import asyncio
import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest

from mosaic.core.config import ExperimentConfig, ModelConfig, ValidationConfig
from mosaic.core.events import (
    ActionExecuted,
    ExperimentStarted,
    ModelCalled,
    SkillExecutionFailed,
    TeacherEscalated,
    read_events,
)
from mosaic.core.models import Action, ModelRequest, ModelResponse
from mosaic.core.provenance import configuration_sha256
from mosaic.environments.fake import FakeEnvironment
from mosaic.providers.base import ModelProvider
from mosaic.providers.fake import FakeModelProvider
from mosaic.skills.models import (
    CandidateSkill,
    SkillProvenance,
    SkillScope,
    VerifiedSkill,
    candidate_content_sha256,
    project_observation,
)
from mosaic.skills.validation import SkillValidator


def fixture_skill(procedure: tuple[Action, ...] | None = None) -> CandidateSkill:
    # A fixed successful visible trajectory, not evidence of model learning.
    environment = FakeEnvironment()
    observation = environment.reset(42, "repair_pump")
    fields = ("position", "part_position", "has_part", "pump_repaired")
    initiation = project_observation(observation.text, fields)
    scope = SkillScope(environment="fake", revision="1", task="repair_pump")
    actions = procedure or (
        *((Action(name="move_east"),) * 7),
        Action(name="collect"),
        *((Action(name="move_west"),) * 7),
        Action(name="repair"),
    )
    tools = tuple(dict.fromkeys(action.name for action in actions))
    digest = candidate_content_sha256(scope, fields, initiation, actions, tools, scope.task)
    return CandidateSkill(
        skill_id=f"skill-{digest[:16]}",
        scope=scope,
        observation_fields=fields,
        initiation_json=initiation,
        procedure=actions,
        allowed_tools=tools,
        termination_goal=scope.task,
        provenance=SkillProvenance(
            source_episode="teacher-source-episode",
            source_agent="agent_0",
            source_seed=42,
            source_provider="fake",
            source_model="source-teacher",
            trace_sha256="a" * 64,
            config_sha256="b" * 64,
            mosaic_commit=None,
        ),
        content_sha256=digest,
    )


def weak_config(
    base: ExperimentConfig,
    policy: Literal["follow_skill", "repair_pump", "first_available"] = "follow_skill",
) -> ExperimentConfig:
    data = base.model_dump(mode="json")
    data["model"]["policy"] = policy
    return ExperimentConfig.model_validate_json(json.dumps(data))


class RecordingProvider(FakeModelProvider):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.policy, config.failed_calls)
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


def test_every_predeclared_seed_is_retained_with_fresh_weak_only_runtime(
    routing_config: ExperimentConfig, tmp_path: Path
) -> None:
    base = weak_config(routing_config)
    before = base.model_dump_json()
    skill = fixture_skill()
    environments: list[FakeEnvironment] = []
    providers: list[RecordingProvider] = []
    configured_trials: list[ExperimentConfig] = []

    def environment_factory(config: ExperimentConfig) -> FakeEnvironment:
        assert config.teacher is None and config.routing is None
        assert config.skills is not None
        assert config.skills.mode == "validation" and config.skills.playbook_path is None
        configured_trials.append(config)
        environment = FakeEnvironment(config.agent.agent_id)
        environments.append(environment)
        return environment

    def weak_factory(config: ModelConfig) -> RecordingProvider:
        assert config == base.model
        provider = RecordingProvider(config)
        providers.append(provider)
        return provider

    output = tmp_path / "validation"
    validation = asyncio.run(
        SkillValidator(environment_factory, weak_factory).validate(
            skill, base, ValidationConfig(seeds=(43, 42), max_steps=32), output
        )
    )
    assert [trial.seed for trial in validation.trials] == [43, 42]
    assert [trial.status for trial in validation.trials] == ["failed", "succeeded"]
    assert [trial.budget.weak.model_calls for trial in validation.trials] == [0, 16]
    assert not validation.passed
    assert len(environments) == len(providers) == len(configured_trials) == 2
    assert environments[0] is not environments[1] and providers[0] is not providers[1]
    assert providers[0].requests == [] and len(providers[1].requests) == 16
    assert base.model_dump_json() == before
    for trial, config in zip(validation.trials, configured_trials, strict=True):
        trace = output / str(trial.seed) / "events.jsonl"
        events = read_events(trace)
        assert trial.config_sha256 == configuration_sha256(config)
        assert trial.trace_sha256 == hashlib.sha256(trace.read_bytes()).hexdigest()
        assert trial.provider == base.model.provider and trial.model == base.model.model
        assert trial.budget.teacher.model_calls == 0
        assert not any(isinstance(event.payload, TeacherEscalated) for event in events)
        calls = [event.payload for event in events if isinstance(event.payload, ModelCalled)]
        assert all(call.model_role == "weak" and call.request.history == () for call in calls)
        started = next(
            event.payload for event in events if isinstance(event.payload, ExperimentStarted)
        )
        assert started.config.teacher is None and started.manifest.teacher_implementation is None


def test_full_skill_requires_sixteen_accepted_actions_and_verified_completion(
    routing_config: ExperimentConfig, tmp_path: Path
) -> None:
    skill = fixture_skill()
    output = tmp_path / "complete"
    validation = asyncio.run(
        SkillValidator(
            lambda config: FakeEnvironment(config.agent.agent_id),
            lambda config: FakeModelProvider(config.policy),
        ).validate(
            skill, weak_config(routing_config), ValidationConfig(seeds=(42,), max_steps=16), output
        )
    )
    assert validation.passed
    trial = validation.trials[0]
    assert trial.status == "succeeded" and trial.skill_completed
    assert trial.budget.weak.model_calls == trial.budget.environment_actions == 16
    assert trial.budget.teacher.model_calls == 0
    events = read_events(output / "42" / "events.jsonl")
    executed = [event.payload for event in events if isinstance(event.payload, ActionExecuted)]
    assert tuple(event.action for event in executed) == skill.procedure
    assert all(event.result.accepted for event in executed)
    assert VerifiedSkill(candidate=skill, validation=validation).status == "verified"


@pytest.mark.parametrize(
    "kind,steps,policy,reason,calls,completed",
    [
        ("partial", 15, "follow_skill", "max_steps", 15, False),
        ("short", 32, "follow_skill", "skill_termination_unverified", 1, False),
        ("divergent", 32, "repair_pump", "skill_execution_failed", 1, False),
        ("ignores", 32, "first_available", "skill_execution_failed", 1, False),
        ("illegal", 32, "follow_skill", "skill_execution_failed", 1, False),
    ],
)
def test_partial_unverified_divergent_and_illegal_skills_cannot_pass(
    routing_config: ExperimentConfig,
    tmp_path: Path,
    kind: str,
    steps: int,
    policy: Literal["follow_skill", "repair_pump", "first_available"],
    reason: str,
    calls: int,
    completed: bool,
) -> None:
    normal = fixture_skill()
    procedure = normal.procedure
    if kind == "short":
        procedure = (Action(name="wait"),)
    elif kind == "divergent":
        procedure = (Action(name="wait"), *procedure)
    elif kind == "illegal":
        procedure = (Action(name="collect"),)
    skill = fixture_skill(procedure)
    output = tmp_path / kind
    validation = asyncio.run(
        SkillValidator(
            lambda config: FakeEnvironment(config.agent.agent_id),
            lambda config: FakeModelProvider(config.policy),
        ).validate(
            skill,
            weak_config(routing_config, policy),
            ValidationConfig(seeds=(42,), max_steps=steps),
            output,
        )
    )
    trial = validation.trials[0]
    assert not validation.passed
    assert trial.status == "failed" and trial.reason == reason
    assert trial.skill_completed == completed
    assert trial.budget.weak.model_calls == calls and trial.budget.teacher.model_calls == 0
    assert trial.budget.skill_execution_failures == 1
    with pytest.raises(ValueError):
        VerifiedSkill(candidate=skill, validation=validation)
    if not completed:
        assert any(
            isinstance(event.payload, SkillExecutionFailed)
            for event in read_events(output / "42" / "events.jsonl")
        )


class FailingProvider:
    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise RuntimeError("Weak model transport error")


def test_weak_transport_error_is_counted_and_does_not_skip_the_next_trial(
    routing_config: ExperimentConfig, tmp_path: Path
) -> None:
    providers: list[ModelProvider] = []

    def weak_factory(config: ModelConfig) -> ModelProvider:
        provider: ModelProvider = (
            FailingProvider() if not providers else FakeModelProvider(config.policy)
        )
        providers.append(provider)
        return provider

    validation = asyncio.run(
        SkillValidator(
            lambda config: FakeEnvironment(config.agent.agent_id), weak_factory
        ).validate(
            fixture_skill(),
            weak_config(routing_config),
            ValidationConfig(seeds=(42, 20), max_steps=32),
            tmp_path / "errors",
        )
    )
    assert [trial.status for trial in validation.trials] == ["error", "succeeded"]
    failed = validation.trials[0]
    assert failed.budget.weak.model_calls == 1
    assert failed.budget.weak.input_tokens is None and failed.budget.weak.output_tokens is None
    assert failed.budget.teacher.model_calls == 0
    assert len(providers) == 2 and not validation.passed


def test_existing_validation_evidence_is_never_overwritten(
    routing_config: ExperimentConfig, tmp_path: Path
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("prior evidence", encoding="utf-8")
    validator = SkillValidator(
        lambda config: FakeEnvironment(config.agent.agent_id),
        lambda config: FakeModelProvider(config.policy),
    )
    with pytest.raises(FileExistsError):
        asyncio.run(
            validator.validate(
                fixture_skill(),
                weak_config(routing_config),
                ValidationConfig(seeds=(42,), max_steps=32),
                output,
            )
        )
    assert sentinel.read_text(encoding="utf-8") == "prior evidence"


def test_completed_trial_costs_survive_later_setup_failure_and_reset_before_next_validation(
    routing_config: ExperimentConfig, tmp_path: Path
) -> None:
    factories_called = 0

    def environment_factory(config: ExperimentConfig) -> FakeEnvironment:
        nonlocal factories_called
        factories_called += 1
        if factories_called > 1:
            raise RuntimeError("Injected later environment setup failure")
        return FakeEnvironment(config.agent.agent_id)

    validator = SkillValidator(environment_factory, lambda config: FakeModelProvider(config.policy))
    settings = ValidationConfig(seeds=(42, 20), max_steps=32)
    with pytest.raises(RuntimeError, match="setup failure"):
        asyncio.run(
            validator.validate(
                fixture_skill(), weak_config(routing_config), settings, tmp_path / "first"
            )
        )
    assert len(validator.completed_runs) == 1
    assert validator.completed_runs[0].summary.budget.weak.model_calls == 16
    assert validator.completed_runs[0].trace_path.exists()
    with pytest.raises(RuntimeError, match="setup failure"):
        asyncio.run(
            validator.validate(
                fixture_skill(), weak_config(routing_config), settings, tmp_path / "second"
            )
        )
    assert not validator.completed_runs
