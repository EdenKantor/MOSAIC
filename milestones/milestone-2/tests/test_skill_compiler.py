import asyncio
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mosaic.core.config import CompilationConfig, ExperimentConfig
from mosaic.core.events import ActionExecuted, read_events
from mosaic.core.models import (
    EnvironmentSnapshot,
    ModelRequest,
    ModelResponse,
    Observation,
    VerificationResult,
)
from mosaic.core.teacher import TeacherCapability
from mosaic.environments.fake import FakeEnvironment
from mosaic.experiments.runner import ExperimentRunner, RunArtifacts
from mosaic.providers.fake import FakeModelProvider
from mosaic.skills.compiler import compile_candidate, extract_teacher_trajectory

FIELDS = ("position", "part_position", "has_part", "pump_repaired")


class CountingTeacher(FakeModelProvider):
    def __init__(self) -> None:
        super().__init__("repair_pump")
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


def source_run(
    config: ExperimentConfig,
    output: Path,
    *,
    environment: FakeEnvironment | None = None,
    teacher: CountingTeacher | None = None,
) -> RunArtifacts:
    capability = None
    if config.teacher is not None:
        capability = TeacherCapability(teacher or CountingTeacher(), config.teacher.access)
    return asyncio.run(
        ExperimentRunner(
            environment or FakeEnvironment(config.agent.agent_id),
            FakeModelProvider(config.model.policy, config.model.failed_calls),
            capability,
        ).run(config, output)
    )


def changed(config: ExperimentConfig, **changes: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate_json(
        json.dumps({**config.model_dump(mode="json"), **changes})
    )


@pytest.fixture
def successful_source(routing_config: ExperimentConfig, tmp_path: Path) -> RunArtifacts:
    result = source_run(routing_config, tmp_path / "source")
    assert result.summary.status == "succeeded"
    assert result.summary.budget.teacher.model_calls == 16
    return result


def test_compilation_uses_only_actual_accepted_teacher_actions(
    successful_source: RunArtifacts,
) -> None:
    source = extract_teacher_trajectory(successful_source)
    events = read_events(successful_source.trace_path)
    accepted = [
        event.payload
        for event in events
        if isinstance(event.payload, ActionExecuted) and event.payload.model_role == "teacher"
    ]
    candidate = compile_candidate(source, CompilationConfig(observation_fields=FIELDS))
    assert candidate.procedure == tuple(event.action for event in accepted)
    assert len(candidate.procedure) == 16
    assert "wait" not in candidate.allowed_tools
    assert candidate.provenance.source_episode == successful_source.summary.episode_id
    assert (
        candidate.provenance.trace_sha256
        == hashlib.sha256(successful_source.trace_path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("kind", ["weak_only", "weak_wins_before_teacher", "failed_source"])
def test_unverified_or_non_teacher_sources_are_rejected(
    routing_config: ExperimentConfig, tmp_path: Path, kind: str
) -> None:
    config = routing_config
    if kind == "failed_source":
        config = changed(config, max_steps=2)
    else:
        data = config.model_dump(mode="json")
        data["model"]["policy"] = "repair_pump"
        if kind == "weak_only":
            data.update(teacher=None, routing=None)
        else:
            data["routing"]["attempt_steps"] = 20
        config = ExperimentConfig.model_validate_json(json.dumps(data))
    result = source_run(config, tmp_path / kind)
    if kind != "failed_source":
        assert result.summary.status == "succeeded"
        assert result.summary.budget.teacher.model_calls == 0
    with pytest.raises(ValueError):
        extract_teacher_trajectory(result)


class HiddenStateEnvironment(FakeEnvironment):
    def observe(self, agent_id: str) -> Observation:
        self._check_agent(agent_id)
        return Observation(agent_id=agent_id, text=super().snapshot().state_json)

    def snapshot(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            state_json=json.dumps(
                {
                    **json.loads(super().snapshot().state_json),
                    "private": "HIDDEN_SNAPSHOT_CANARY",
                }
            )
        )

    def verify(self, goal_id: str) -> VerificationResult:
        return super().verify(goal_id).model_copy(update={"evidence": "HIDDEN_VERIFIER_CANARY"})


def test_sanitized_compiler_input_excludes_private_audit_data_and_invokes_no_teacher(
    routing_config: ExperimentConfig, tmp_path: Path
) -> None:
    teacher = CountingTeacher()
    result = source_run(
        routing_config, tmp_path / "hidden", environment=HiddenStateEnvironment(), teacher=teacher
    )
    raw = result.trace_path.read_text(encoding="utf-8")
    assert "HIDDEN_SNAPSHOT_CANARY" in raw and "HIDDEN_VERIFIER_CANARY" in raw
    calls_before = len(teacher.requests)
    requests_before = tuple(teacher.requests)
    projection = extract_teacher_trajectory(result)
    candidate = compile_candidate(projection, CompilationConfig(observation_fields=FIELDS))
    assert len(teacher.requests) == calls_before == 16
    assert tuple(teacher.requests) == requests_before
    for serialized in (projection.model_dump_json(), candidate.model_dump_json()):
        assert "HIDDEN_SNAPSHOT_CANARY" not in serialized
        assert "HIDDEN_VERIFIER_CANARY" not in serialized
    assert all(
        "HIDDEN_SNAPSHOT_CANARY" not in request.model_dump_json()
        and "HIDDEN_VERIFIER_CANARY" not in request.model_dump_json()
        for request in teacher.requests
    )


def test_identical_content_has_same_id_despite_different_source_provenance(
    routing_config: ExperimentConfig, tmp_path: Path
) -> None:
    first = source_run(routing_config, tmp_path / "first")
    second = source_run(routing_config, tmp_path / "second")
    settings = CompilationConfig(observation_fields=FIELDS)
    one = compile_candidate(extract_teacher_trajectory(first), settings)
    two = compile_candidate(extract_teacher_trajectory(second), settings)
    assert one.skill_id == two.skill_id and one.content_sha256 == two.content_sha256
    assert one.provenance.source_episode != two.provenance.source_episode
    assert one.provenance.trace_sha256 != two.provenance.trace_sha256


def test_bounded_contextual_sequence_matches_seen_conditions_and_excludes_changed_layout(
    successful_source: RunArtifacts,
) -> None:
    trajectory = extract_teacher_trajectory(successful_source)
    with pytest.raises(ValueError, match="bound"):
        compile_candidate(trajectory, CompilationConfig(observation_fields=FIELDS, max_actions=15))
    candidate = compile_candidate(
        trajectory, CompilationConfig(observation_fields=FIELDS, max_actions=16)
    )
    environment = FakeEnvironment()
    assert candidate.matches(candidate.scope, environment.reset(42, "repair_pump"))
    assert not candidate.matches(candidate.scope, environment.reset(43, "repair_pump"))
    assert "steps" not in candidate.initiation_json


def test_teacher_suffix_preserves_the_actual_partial_progress_precondition(
    routing_config: ExperimentConfig, tmp_path: Path
) -> None:
    data = routing_config.model_dump(mode="json")
    data["model"]["policy"] = "repair_pump"
    result = source_run(
        ExperimentConfig.model_validate_json(json.dumps(data)), tmp_path / "partial"
    )
    assert result.summary.status == "succeeded" and result.summary.budget.teacher.model_calls == 14
    trajectory = extract_teacher_trajectory(result)
    candidate = compile_candidate(trajectory, CompilationConfig(observation_fields=FIELDS))
    assert candidate.matches(candidate.scope, trajectory.transitions[0].observation)
    assert not candidate.matches(candidate.scope, FakeEnvironment().reset(42, "repair_pump"))


class InitiallyInvalidTeacher(CountingTeacher):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        first = not self.requests
        response = await super().generate(request)
        return (
            response.model_copy(update={"text": '{"name":"not_advertised"}'}) if first else response
        )


def test_teacher_rejection_disqualifies_an_otherwise_successful_source(
    routing_config: ExperimentConfig, tmp_path: Path
) -> None:
    result = source_run(routing_config, tmp_path / "rejection", teacher=InitiallyInvalidTeacher())
    assert result.summary.status == "succeeded" and result.summary.budget.teacher.model_calls == 17
    with pytest.raises(ValueError):
        extract_teacher_trajectory(result)


def tampered_source(
    source: RunArtifacts,
    output: Path,
    mutation: Callable[[list[dict[str, Any]]], object],
) -> RunArtifacts:
    events = [
        json.loads(line) for line in source.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    mutation(events)
    output.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return RunArtifacts(output, source.summary_path, source.summary)


@pytest.mark.parametrize(
    "kind",
    [
        "call_provider",
        "request_role",
        "request_model",
        "request_goal",
        "request_agent",
        "event_agent",
        "event_experiment",
        "request_step",
        "call_index",
        "response_model",
        "response_provider",
        "response_role",
        "response_action",
        "unobserved_request",
        "unobserved_actions",
        "result_agent",
        "false_task_success",
        "wrong_goal_success",
        "duplicate_execution",
    ],
)
def test_source_identity_and_action_evidence_fail_closed_when_trace_is_inconsistent(
    successful_source: RunArtifacts, tmp_path: Path, kind: str
) -> None:
    def mutate(events: list[dict[str, Any]]) -> None:
        call_event = next(
            event
            for event in events
            if event["event_type"] == "ModelCalled" and event["payload"]["model_role"] == "teacher"
        )
        response_event = next(
            event
            for event in events
            if event["event_type"] == "ModelResponded"
            and event["payload"]["model_role"] == "teacher"
        )
        call, request, response = (
            call_event["payload"],
            call_event["payload"]["request"],
            response_event["payload"]["response"],
        )
        if kind == "call_provider":
            call["provider"] = "another-provider"
        elif kind.startswith("request_"):
            field = kind.removeprefix("request_")
            if field == "agent":
                request["observation"]["agent_id"] = "another-agent"
            elif field == "step":
                request["step"] = 99
            else:
                request[{"role": "model_role", "model": "model", "goal": "goal"}[field]] = (
                    "weak" if field == "role" else "wrong"
                )
        elif kind == "event_agent":
            call_event["agent_id"] = "another-agent"
        elif kind == "event_experiment":
            call_event["experiment_id"] = "another-experiment"
        elif kind == "call_index":
            call["call_index"] = 999
        elif kind == "response_model":
            response["model"] = "another-model"
        elif kind == "response_provider":
            response["provider"] = "another-provider"
        elif kind == "response_role":
            response_event["payload"]["model_role"] = "weak"
        elif kind == "response_action":
            response["text"] = '{"name":"wait"}'
        elif kind == "unobserved_request":
            request["observation"]["text"] = '{"position":99}'
        elif kind == "unobserved_actions":
            request["available_actions"][0]["description"] = "unobserved action description"
        elif kind == "result_agent":
            executed = next(
                event["payload"]
                for event in events
                if event["event_type"] == "ActionExecuted"
                and event["payload"]["model_role"] == "teacher"
            )
            executed["result"]["observation"]["agent_id"] = "another-agent"
        elif kind in {"false_task_success", "wrong_goal_success"}:
            success = next(
                event["payload"] for event in events if event["event_type"] == "TaskSucceeded"
            )
            success["verification"]["succeeded" if kind == "false_task_success" else "goal_id"] = (
                False if kind == "false_task_success" else "another-goal"
            )
        elif kind == "duplicate_execution":
            executed = [
                event
                for event in events
                if event["event_type"] == "ActionExecuted"
                and event["payload"]["model_role"] == "teacher"
            ]
            executed[1]["step"] = executed[0]["step"]
            executed[1]["payload"] = executed[0]["payload"]

    altered = tampered_source(successful_source, tmp_path / f"{kind}.jsonl", mutate)
    with pytest.raises(ValueError):
        extract_teacher_trajectory(altered)


def test_trace_must_be_complete_contiguous_and_have_matching_artifact_summary(
    successful_source: RunArtifacts, tmp_path: Path
) -> None:
    truncated = tampered_source(
        successful_source, tmp_path / "truncated.jsonl", lambda events: events.pop()
    )
    with pytest.raises(ValueError):
        extract_teacher_trajectory(truncated)
    skipped = tampered_source(
        successful_source, tmp_path / "skipped.jsonl", lambda events: events.pop(10)
    )
    with pytest.raises(ValueError):
        extract_teacher_trajectory(skipped)
    different_summary = successful_source.summary.model_copy(update={"reason": "mismatched"})
    with pytest.raises(ValueError):
        extract_teacher_trajectory(
            RunArtifacts(
                successful_source.trace_path, successful_source.summary_path, different_summary
            )
        )


def test_uninterrupted_teacher_sequence_rejects_an_interleaved_weak_execution(
    successful_source: RunArtifacts, tmp_path: Path
) -> None:
    def insert_weak(events: list[dict[str, Any]]) -> None:
        weak = next(
            event
            for event in events
            if event["event_type"] == "ActionExecuted" and event["payload"]["model_role"] == "weak"
        )
        second_teacher = [
            i
            for i, event in enumerate(events)
            if event["event_type"] == "ModelCalled" and event["payload"]["model_role"] == "teacher"
        ][1]
        events.insert(second_teacher, json.loads(json.dumps(weak)))
        for sequence, event in enumerate(events):
            event["sequence"] = sequence

    altered = tampered_source(successful_source, tmp_path / "interleaved.jsonl", insert_weak)
    with pytest.raises(ValueError):
        extract_teacher_trajectory(altered)
