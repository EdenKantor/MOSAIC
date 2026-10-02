import asyncio
import json
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import ValidationError
from test_experiments import semantic

from mosaic.core.config import ExperimentConfig, TeacherAccess, load_config
from mosaic.core.events import (
    ActionExecuted,
    ActionRejected,
    ModelCalled,
    ModelCallFailed,
    ModelResponded,
    TeacherEscalated,
    TeacherEscalationEligible,
    TeacherInvocationBlocked,
    WeakRecoveryStarted,
    read_events,
)
from mosaic.core.models import (
    Action,
    EnvironmentSnapshot,
    ModelRequest,
    ModelResponse,
    Observation,
    StepResult,
    VerificationResult,
)
from mosaic.core.teacher import TeacherCapability, TeacherUnavailable
from mosaic.environments.fake import FakeEnvironment
from mosaic.experiments.runner import ExperimentRunner, RunArtifacts
from mosaic.providers.base import ModelProvider
from mosaic.providers.fake import FakeModelProvider


def changed(config: ExperimentConfig, **changes: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate_json(json.dumps({**config.model_dump(), **changes}))


def run_routing(
    config: ExperimentConfig,
    output: Path,
    *,
    weak: ModelProvider | None = None,
    teacher: ModelProvider | None = None,
    environment: FakeEnvironment | None = None,
) -> RunArtifacts:
    assert config.teacher is not None
    weak_provider = weak or FakeModelProvider(config.model.policy, config.model.failed_calls)
    teacher_provider = teacher or FakeModelProvider(config.teacher.model.policy)
    runner = ExperimentRunner(
        environment or FakeEnvironment(config.agent.agent_id),
        weak_provider,
        TeacherCapability(teacher_provider, config.teacher.access),
    )
    return asyncio.run(runner.run(config, output))


def test_no_premature_escalation_and_objective_window_failure(
    routing_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    result = run_routing(routing_config, tmp_path / "run")
    events = read_events(result.trace_path)
    first_teacher = next(
        i
        for i, e in enumerate(events)
        if isinstance(e.payload, ModelCalled) and e.payload.model_role == "teacher"
    )
    weak_calls = [e.payload for e in events[:first_teacher] if isinstance(e.payload, ModelCalled)]
    assert len(weak_calls) == 2 and all(e.model_role == "weak" for e in weak_calls)
    assert sum(isinstance(e.payload, WeakRecoveryStarted) for e in events[:first_teacher]) == 1
    eligible = next(e.payload for e in events if isinstance(e.payload, TeacherEscalationEligible))
    assert eligible.reason == "weak_window_exhausted_without_goal"
    assert eligible.recoveries_started == 1 and eligible.phase_steps == 1
    assert result.summary.status == "succeeded"
    assert result.summary.budget.weak.model_calls == 2
    assert result.summary.budget.teacher.model_calls == 16
    assert result.summary.budget.environment_actions == 18
    assert result.summary.escalations == 1
    # Accepted wait actions are unsuccessful within the finite success window.
    weak_actions = [
        e.payload
        for e in events
        if isinstance(e.payload, ActionExecuted) and e.payload.model_role == "weak"
    ]
    assert all(e.action.name == "wait" for e in weak_actions)


@pytest.mark.parametrize(
    "filename,weak_calls,recoveries",
    [
        ("weak-success.yaml", 16, 0),
        ("recovery-success.yaml", 17, 1),
    ],
)
def test_weak_and_recovery_success_never_call_teacher(
    tmp_path: Path,
    filename: str,
    weak_calls: int,
    recoveries: int,
) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    result = run_routing(config, tmp_path / "run")
    events = read_events(result.trace_path)
    assert result.summary.status == "succeeded"
    assert result.summary.budget.weak.model_calls == weak_calls
    assert result.summary.budget.teacher.model_calls == 0
    assert result.summary.escalations == 0
    assert sum(isinstance(e.payload, WeakRecoveryStarted) for e in events) == recoveries
    assert not any(isinstance(e.payload, TeacherEscalationEligible) for e in events)


class CountingTeacher(FakeModelProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return await super().generate(request)


def test_withdrawn_teacher_is_blocked_before_dispatch(
    routing_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    assert routing_config.teacher is not None
    config = changed(
        routing_config, teacher={**routing_config.teacher.model_dump(), "access": "withdrawn"}
    )
    teacher = CountingTeacher()
    result = run_routing(config, tmp_path / "run", teacher=teacher)
    assert teacher.calls == 0
    assert result.summary.status == "failed" and result.summary.reason == "teacher_withdrawn"
    assert result.summary.budget.teacher.model_calls == 0
    assert result.summary.blocked_escalations == 1 and result.summary.escalations == 0
    events = read_events(result.trace_path)
    assert any(isinstance(e.payload, TeacherEscalationEligible) for e in events)
    assert any(isinstance(e.payload, TeacherInvocationBlocked) for e in events)
    assert not any(isinstance(e.payload, TeacherEscalated) for e in events)


def test_capability_itself_prevents_teacher_calls_after_withdrawal(
    routing_config: ExperimentConfig,
) -> None:
    teacher = CountingTeacher()
    capability = TeacherCapability(teacher)
    capability.withdraw()
    assert capability.access == TeacherAccess.WITHDRAWN
    from mosaic.core.agent import Agent

    env = FakeEnvironment()
    observation = env.reset(42, "repair_pump")
    request = Agent(routing_config).request(observation, env.available_actions("agent_0"))
    with pytest.raises(TeacherUnavailable):
        asyncio.run(capability.generate(request))
    assert teacher.calls == 0


class WithdrawingTeacher(CountingTeacher):
    capability: TeacherCapability

    async def generate(self, request: ModelRequest) -> ModelResponse:
        response = await super().generate(request)
        self.capability.withdraw()
        return response


def test_withdrawal_during_takeover_blocks_the_next_invocation(
    routing_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    teacher = WithdrawingTeacher()
    capability = TeacherCapability(teacher)
    teacher.capability = capability
    runner = ExperimentRunner(FakeEnvironment(), FakeModelProvider("first_available"), capability)
    result = asyncio.run(runner.run(routing_config, tmp_path / "run"))
    assert teacher.calls == result.summary.budget.teacher.model_calls == 1
    assert result.summary.status == "failed" and result.summary.reason == "teacher_withdrawn"
    assert result.summary.escalations == result.summary.blocked_escalations == 1
    assert any(
        isinstance(e.payload, TeacherInvocationBlocked) for e in read_events(result.trace_path)
    )


def test_role_budgets_reconcile_every_invocation_and_latency(
    routing_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    result = run_routing(routing_config, tmp_path / "run")
    events = read_events(result.trace_path)
    for role, budget in (
        ("weak", result.summary.budget.weak),
        ("teacher", result.summary.budget.teacher),
    ):
        calls = [
            e.payload
            for e in events
            if isinstance(e.payload, ModelCalled) and e.payload.model_role == role
        ]
        responses = [
            e.payload
            for e in events
            if isinstance(e.payload, ModelResponded) and e.payload.model_role == role
        ]
        assert budget.model_calls == len(calls) == len(responses)
        assert budget.input_tokens == sum(e.response.usage.input_tokens or 0 for e in responses)
        assert budget.output_tokens == sum(e.response.usage.output_tokens or 0 for e in responses)
        assert budget.wall_clock_ms == pytest.approx(sum(e.latency_ms for e in responses))
        assert budget.synthetic_usage_calls == budget.model_calls
        assert all(e.request.model_role == role and e.provider == "fake" for e in calls)
    assert result.summary.total_model_calls == result.summary.budget.model_calls == 18
    assert result.summary.teacher_call_fraction == pytest.approx(16 / 18)


def test_routing_is_deterministic(routing_config: ExperimentConfig, tmp_path: Path) -> None:
    first = read_events(run_routing(routing_config, tmp_path / "first").trace_path)
    second = read_events(run_routing(routing_config, tmp_path / "second").trace_path)
    assert [semantic(e.model_dump(mode="json")) for e in first] == [
        semantic(e.model_dump(mode="json")) for e in second
    ]


class RecordingProvider(FakeModelProvider):
    def __init__(
        self,
        policy: Literal["repair_pump", "first_available", "wait_then_repair"],
        mark_teacher: bool = False,
    ) -> None:
        super().__init__(policy)
        self.requests: list[ModelRequest] = []
        self.mark_teacher = mark_teacher

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        response = await super().generate(request)
        return (
            response.model_copy(update={"text": response.text + " " * 37})
            if self.mark_teacher
            else response
        )


def test_no_cross_episode_teacher_memory_even_with_reused_runner(
    routing_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    weak, teacher = RecordingProvider("first_available"), RecordingProvider("repair_pump", True)
    runner = ExperimentRunner(FakeEnvironment(), weak, TeacherCapability(teacher))
    first = asyncio.run(runner.run(routing_config, tmp_path / "first"))
    first_weak, first_teacher = tuple(weak.requests), tuple(teacher.requests)
    assert any(" " * 37 in entry.proposal for req in first_teacher for entry in req.history)
    second = asyncio.run(runner.run(routing_config, tmp_path / "second"))
    second_weak, second_teacher = tuple(weak.requests[2:]), tuple(teacher.requests[16:])
    assert first.summary.status == second.summary.status == "succeeded"
    assert first_weak == second_weak and first_teacher == second_teacher
    assert second_weak[0].history == ()
    assert all(entry.model_role == "weak" for entry in second_teacher[0].history)
    assert not any(" " * 37 in entry.proposal for req in second_weak for entry in req.history)


class HiddenStateEnvironment(FakeEnvironment):
    def observe(self, agent_id: str) -> Observation:
        self._check_agent(agent_id)
        return Observation(agent_id=agent_id, text=super().snapshot().state_json)

    def snapshot(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            state_json=json.dumps(
                {
                    **json.loads(super().snapshot().state_json),
                    "private": "HIDDEN_STATE_CANARY",
                }
            )
        )

    def verify(self, goal_id: str) -> VerificationResult:
        result = super().verify(goal_id)
        return result.model_copy(update={"evidence": "PRIVATE_VERIFIER_CANARY"})


def test_teacher_receives_visible_context_without_snapshot_or_verifier_internals(
    routing_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    teacher = RecordingProvider("repair_pump")
    result = run_routing(
        routing_config, tmp_path / "run", teacher=teacher, environment=HiddenStateEnvironment()
    )
    assert result.summary.status == "succeeded"
    assert teacher.requests[0].goal == "repair_pump"
    assert len(teacher.requests[0].history) == 2
    assert "HIDDEN_STATE_CANARY" in result.trace_path.read_text()
    for request in teacher.requests:
        assert "HIDDEN_STATE_CANARY" not in request.model_dump_json()
        assert "PRIVATE_VERIFIER_CANARY" not in request.model_dump_json()
        assert len(request.history) <= 8


class RejectFirstTwoEnvironment(FakeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.rejections = 0

    def reset(self, seed: int, task_id: str) -> Observation:
        self.rejections = 0
        return super().reset(seed, task_id)

    def step(self, agent_id: str, action: Action) -> StepResult:
        if self.rejections < 2:
            self.rejections += 1
            return StepResult(
                observation=self.observe(agent_id),
                accepted=False,
                reason="Objective environment failure",
            )
        return super().step(agent_id, action)


def test_rejected_environment_actions_trigger_bounded_recovery_and_escalation(
    routing_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    assert routing_config.routing is not None
    config = changed(
        routing_config,
        routing={
            **routing_config.routing.model_dump(),
            "attempt_steps": 20,
            "recovery_steps": 20,
            "consecutive_failure_limit": 1,
        },
    )
    result = run_routing(config, tmp_path / "run", environment=RejectFirstTwoEnvironment())
    events = read_events(result.trace_path)
    assert result.summary.status == "succeeded"
    assert result.summary.budget.weak.model_calls == 2
    assert sum(isinstance(e.payload, ActionRejected) and e.payload.dispatched for e in events) == 2
    eligible = next(e.payload for e in events if isinstance(e.payload, TeacherEscalationEligible))
    assert (
        eligible.reason == "consecutive_execution_failures" and eligible.consecutive_failures == 1
    )


def test_total_step_limit_prevents_unaffordable_escalation(
    routing_config: ExperimentConfig, tmp_path: Path
) -> None:
    result = run_routing(changed(routing_config, max_steps=2), tmp_path / "run")
    assert result.summary.status == "failed" and result.summary.reason == "max_steps"
    assert result.summary.budget.teacher.model_calls == 0
    assert not any(
        isinstance(e.payload, TeacherEscalationEligible) for e in read_events(result.trace_path)
    )


class FailingTeacher:
    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise RuntimeError("Teacher transport error")


def test_teacher_transport_failure_counts_unknown_usage_separately(
    routing_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    result = run_routing(routing_config, tmp_path / "run", teacher=FailingTeacher())
    assert result.summary.status == "error"
    assert result.summary.budget.weak.model_calls == 2
    assert result.summary.budget.weak.input_tokens is not None
    assert result.summary.budget.teacher.model_calls == 1
    assert result.summary.budget.teacher.input_tokens is None
    assert result.summary.budget.teacher.output_tokens is None
    assert result.summary.budget.input_tokens is None
    failures = [
        e.payload for e in read_events(result.trace_path) if isinstance(e.payload, ModelCallFailed)
    ]
    assert len(failures) == 1 and failures[0].model_role == "teacher"
    assert result.summary.budget.teacher.wall_clock_ms == failures[0].latency_ms


def test_runtime_requires_configured_teacher_capability(
    routing_config: ExperimentConfig, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="capability"):
        asyncio.run(
            ExperimentRunner(FakeEnvironment(), FakeModelProvider()).run(
                routing_config, tmp_path / "run"
            )
        )


def test_teacher_and_policy_cannot_be_silently_omitted(routing_config: ExperimentConfig) -> None:
    with pytest.raises(ValidationError):
        changed(routing_config, teacher=None)
    with pytest.raises(ValidationError):
        changed(routing_config, routing=None)
