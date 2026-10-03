import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from mosaic.core.agent import SYSTEM_PROMPT, Agent
from mosaic.core.budget import BudgetLedger
from mosaic.core.config import ExperimentConfig, ModelConfig
from mosaic.core.events import (
    ActionDispatched,
    ActionExecuted,
    ActionProposed,
    ActionRejected,
    EnvironmentChanged,
    EpisodeFinished,
    EpisodeStarted,
    EventRecorder,
    ExperimentFinished,
    ExperimentStarted,
    ModelCalled,
    ModelCallFailed,
    ModelResponded,
    ObservationReceived,
    RunErrored,
    RunSummary,
    TaskFailed,
    TaskSucceeded,
    TeacherEscalated,
    TeacherEscalationEligible,
    TeacherInvocationBlocked,
    VerificationPerformed,
    WeakAttemptStarted,
    WeakRecoveryStarted,
)
from mosaic.core.models import (
    AttemptFeedback,
    EnvironmentSnapshot,
    ModelRequest,
    ModelResponse,
    ModelRole,
    VerificationResult,
)
from mosaic.core.provenance import manifest
from mosaic.core.routing import BoundedFailurePolicy, EscalationPolicy, RoutingState
from mosaic.core.teacher import TeacherCapability, TeacherUnavailable
from mosaic.environments.base import EnvironmentAdapter
from mosaic.providers.base import ModelProvider
from mosaic.providers.http import ProviderError


@dataclass(frozen=True)
class RunArtifacts:
    trace_path: Path
    summary_path: Path
    summary: RunSummary


class InferenceBatchStopped(RuntimeError):
    """A proactive budget guard blocked dispatch; no physical request was attempted."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


async def _invoke(
    provider: ModelProvider,
    request: ModelRequest,
    identity: ModelConfig,
    role: ModelRole,
    ledger: BudgetLedger,
    recorder: EventRecorder,
    step: int,
) -> ModelResponse:
    guard = getattr(provider, "before_dispatch", None)
    if callable(guard):
        await guard(request)
    call_index = ledger.model_called(role)
    recorder.record(
        ModelCalled(
            call_index=call_index, request=request, model_role=role, provider=identity.provider
        ),
        step,
    )
    called = time.perf_counter()
    try:
        response = await provider.generate(request)
    except Exception as error:
        latency_ms = (time.perf_counter() - called) * 1000
        ledger.model_finished(call_index, latency_ms)
        recorder.record(
            ModelCallFailed(
                call_index=call_index,
                model_role=role,
                provider=identity.provider,
                model=identity.model,
                latency_ms=latency_ms,
                error_type=type(error).__name__,
                message="Provider inference request failed",
                provider_category=error.category if isinstance(error, ProviderError) else None,
                http_status=error.http_status if isinstance(error, ProviderError) else None,
                stop_batch=error.stop_batch if isinstance(error, ProviderError) else False,
            ),
            step,
        )
        raise
    latency_ms = (time.perf_counter() - called) * 1000
    ledger.model_finished(call_index, latency_ms)
    ledger.model_responded(call_index, response.usage)
    recorder.record(
        ModelResponded(
            call_index=call_index, response=response, latency_ms=latency_ms, model_role=role
        ),
        step,
    )
    if response.provider != identity.provider or response.model != identity.model:
        raise ValueError("Provider response identity does not match immutable configuration")
    return response


class ExperimentRunner:
    def __init__(
        self,
        environment: EnvironmentAdapter,
        provider: ModelProvider,
        teacher: TeacherCapability | None = None,
        routing_policy: EscalationPolicy | None = None,
    ) -> None:
        self.environment = environment
        self.provider = provider
        self.teacher = teacher
        self.routing_policy = routing_policy

    async def run(self, config: ExperimentConfig, output_dir: Path) -> RunArtifacts:
        if (config.teacher is None) != (self.teacher is None):
            raise ValueError("Runtime teacher capability must match experiment configuration")
        if config.teacher and self.teacher and config.teacher.access != self.teacher.access:
            raise ValueError("Initial runtime teacher availability must match configuration")
        if config.routing is None and self.routing_policy is not None:
            raise ValueError("A routing policy requires explicit routing configuration")
        policy = self.routing_policy
        if config.routing and policy is None:
            policy = BoundedFailurePolicy(config.routing)
        if isinstance(policy, BoundedFailurePolicy) and policy.config != config.routing:
            raise ValueError("Routing policy limits must match immutable configuration")

        output_dir.mkdir(parents=True, exist_ok=False)
        episode_id = str(uuid4())
        trace_path, summary_path = output_dir / "events.jsonl", output_dir / "summary.json"
        recorder = EventRecorder(
            trace_path, config.experiment_id, episode_id, config.agent.agent_id
        )
        ledger, agent = BudgetLedger(), Agent(config)
        started = time.perf_counter()
        status: Literal["succeeded", "failed", "error"] = "failed"
        reason, stage = "max_steps", "provenance"
        verification: VerificationResult | None = None
        snapshot: EnvironmentSnapshot | None = None
        steps = escalations = blocked_escalations = 0
        invalid_json = invalid_action = provider_failures = 0
        provider_failure_category: str | None = None
        provider_http_status: int | None = None
        provider_stop_batch = False
        batch_stop_reason: str | None = None
        role: ModelRole = "weak"
        phase: Literal["attempt", "recovery"] = "attempt"
        phase_steps = recoveries_started = consecutive_failures = 0
        history: tuple[AttemptFeedback, ...] = ()
        try:
            recorder.record(
                ExperimentStarted(
                    config=config,
                    manifest=manifest(self.environment, self.provider, self.teacher, policy),
                )
            )
            stage = "reset"
            reset_observation = self.environment.reset(config.seed, config.task)
            if reset_observation.agent_id != config.agent.agent_id:
                raise ValueError("Reset observation belongs to a different agent")
            snapshot = self.environment.snapshot()
            recorder.record(EpisodeStarted(snapshot=snapshot), 0)
            stage = "verify"
            verification = self.environment.verify(config.task)
            if verification.goal_id != config.task:
                raise ValueError("Environment verifier returned a different goal")
            recorder.record(VerificationPerformed(verification=verification), 0)
            if verification.succeeded:
                status, reason = "succeeded", "environment_verified"
            else:
                if config.routing:
                    recorder.record(
                        WeakAttemptStarted(window_steps=config.routing.attempt_steps), 0
                    )
                for step in range(1, config.max_steps + 1):
                    steps = step
                    stage = "observe"
                    observation = self.environment.observe(config.agent.agent_id)
                    if observation.agent_id != config.agent.agent_id:
                        raise ValueError("Environment observation belongs to a different agent")
                    actions = self.environment.available_actions(config.agent.agent_id)
                    recorder.record(
                        ObservationReceived(observation=observation, available_actions=actions),
                        step,
                    )
                    if not actions:
                        reason = "no_available_actions"
                        break
                    identity, active_provider = config.model, self.provider
                    if role == "teacher":
                        assert config.teacher is not None and self.teacher is not None
                        try:
                            self.teacher.require_available()
                        except TeacherUnavailable as error:
                            blocked_escalations += 1
                            recorder.record(
                                TeacherInvocationBlocked(
                                    access=self.teacher.access, reason=str(error)
                                ),
                                step,
                            )
                            reason = "teacher_withdrawn"
                            break
                        identity, active_provider = config.teacher.model, self.teacher
                    call_role = role if config.routing else config.agent_model_role
                    system_prompt = None
                    if config.prompt_protocol == "alem_official_v1":
                        stage = "prompt"
                        prompt_builder = getattr(self.environment, "instruction_prompt", None)
                        if not callable(prompt_builder):
                            raise ValueError(
                                "The reviewed prompt protocol needs an environment rules builder"
                            )
                        rules = prompt_builder()
                        if not isinstance(rules, str) or not rules:
                            raise ValueError(
                                "The environment rules prompt must be a nonempty string"
                            )
                        system_prompt = rules + "\n\n" + SYSTEM_PROMPT
                    request = agent.request(
                        observation,
                        actions,
                        model=identity,
                        model_role=call_role,
                        history=history if config.routing or system_prompt else (),
                        step=step,
                        system_prompt=system_prompt,
                    )
                    stage = "model_generate"
                    response = await _invoke(
                        active_provider, request, identity, call_role, ledger, recorder, step
                    )
                    action = None
                    environment_done = False
                    feedback_observation = observation
                    try:
                        action = agent.parse(response.text, actions)
                    except ValueError as error:
                        if str(error).startswith("Action is not advertised:"):
                            invalid_action += 1
                            feedback = "Action is not advertised"
                        else:
                            invalid_json += 1
                            feedback = "Action JSON or schema is invalid"
                        accepted = False
                        recorder.record(
                            ActionRejected(
                                proposal=response.text,
                                reason=feedback,
                                dispatched=False,
                                model_role=call_role,
                            ),
                            step,
                        )
                    else:
                        recorder.record(ActionProposed(action=action, model_role=call_role), step)
                        stage = "environment_step"
                        ledger.action_called()
                        recorder.record(ActionDispatched(action=action, model_role=call_role), step)
                        result = self.environment.step(config.agent.agent_id, action)
                        if result.observation.agent_id != config.agent.agent_id:
                            raise ValueError("Step observation belongs to a different agent")
                        accepted, feedback, feedback_observation = (
                            result.accepted,
                            result.reason,
                            result.observation,
                        )
                        if result.accepted:
                            recorder.record(
                                ActionExecuted(action=action, result=result, model_role=call_role),
                                step,
                            )
                        else:
                            invalid_action += 1
                            recorder.record(
                                ActionRejected(
                                    proposal=response.text,
                                    reason=result.reason,
                                    dispatched=True,
                                    result=result,
                                    model_role=call_role,
                                ),
                                step,
                            )
                        snapshot = self.environment.snapshot()
                        recorder.record(EnvironmentChanged(snapshot=snapshot), step)
                        stage = "verify"
                        verification = self.environment.verify(config.task)
                        if verification.goal_id != config.task:
                            raise ValueError("Environment verifier returned a different goal")
                        recorder.record(VerificationPerformed(verification=verification), step)
                        environment_done = result.terminated or result.truncated
                        if environment_done:
                            reason = (
                                "environment_terminated"
                                if result.terminated
                                else "environment_truncated"
                            )
                    if config.routing or config.prompt_protocol == "alem_official_v1":
                        limit = (
                            config.routing.history_limit if config.routing else config.history_limit
                        )
                        history = (
                            (
                                *history,
                                AttemptFeedback(
                                    step=step,
                                    model_role=call_role,
                                    proposal=response.text,
                                    action=action,
                                    accepted=accepted,
                                    feedback=feedback,
                                    observation=feedback_observation,
                                ),
                            )[-limit:]
                            if limit
                            else ()
                        )
                    if verification.succeeded:
                        status, reason = "succeeded", "environment_verified"
                        break
                    if environment_done:
                        break
                    if role == "weak" and policy and config.routing:
                        phase_steps += 1
                        consecutive_failures = 0 if accepted else consecutive_failures + 1
                        state = RoutingState(
                            phase=phase,
                            phase_steps=phase_steps,
                            recoveries_started=recoveries_started,
                            consecutive_failures=consecutive_failures,
                            remaining_steps=config.max_steps - step,
                            succeeded=verification.succeeded,
                            environment_done=environment_done,
                        )
                        decision = policy.decide(state)
                        if decision.outcome == "terminate":
                            reason = decision.reason
                            break
                        if decision.outcome == "start_recovery":
                            recoveries_started += 1
                            phase, phase_steps, consecutive_failures = "recovery", 0, 0
                            recorder.record(
                                WeakRecoveryStarted(
                                    recovery_number=recoveries_started,
                                    window_steps=config.routing.recovery_steps,
                                    reason=decision.reason,
                                ),
                                step,
                            )
                        elif decision.outcome == "escalate":
                            recorder.record(
                                TeacherEscalationEligible(
                                    reason=decision.reason,
                                    phase_steps=phase_steps,
                                    recoveries_started=recoveries_started,
                                    consecutive_failures=consecutive_failures,
                                ),
                                step,
                            )
                            assert self.teacher is not None and config.teacher is not None
                            try:
                                self.teacher.require_available()
                            except TeacherUnavailable as error:
                                blocked_escalations += 1
                                recorder.record(
                                    TeacherInvocationBlocked(
                                        access=self.teacher.access, reason=str(error)
                                    ),
                                    step,
                                )
                                reason = "teacher_withdrawn"
                                break
                            escalations += 1
                            role = "teacher"
                            recorder.record(
                                TeacherEscalated(
                                    provider=config.teacher.model.provider,
                                    model=config.teacher.model.model,
                                    access=self.teacher.access,
                                ),
                                step,
                            )
            if status == "succeeded":
                assert verification is not None
                recorder.record(TaskSucceeded(verification=verification), steps)
            else:
                recorder.record(TaskFailed(reason=reason, verification=verification), steps)
        except Exception as error:
            status, reason = "error", f"{stage}: runtime_failure"
            if isinstance(error, InferenceBatchStopped):
                batch_stop_reason = error.reason
                reason = f"inference_batch_stopped: {error.reason}"
            elif stage == "model_generate":
                provider_failures += 1
                if isinstance(error, ProviderError):
                    provider_failure_category = error.category
                    provider_http_status = error.http_status
                    provider_stop_batch = error.stop_batch
                    reason = f"provider_failure: {error.category}"
            recorder.record(
                RunErrored(
                    stage=stage,
                    error_type=type(error).__name__,
                    message="Experiment operation failed",
                ),
                steps,
            )
        finally:
            try:
                budget = ledger.totals((time.perf_counter() - started) * 1000)
                summary = RunSummary(
                    experiment_id=config.experiment_id,
                    episode_id=episode_id,
                    status=status,
                    reason=reason,
                    steps=steps,
                    verification=verification,
                    budget=budget,
                    task=config.task,
                    total_model_calls=budget.model_calls,
                    teacher_call_fraction=budget.teacher.model_calls / budget.model_calls
                    if budget.model_calls
                    else None,
                    escalations=escalations,
                    blocked_escalations=blocked_escalations,
                    invalid_json=invalid_json,
                    invalid_action=invalid_action,
                    provider_failures=provider_failures,
                    provider_failure_category=provider_failure_category,
                    provider_http_status=provider_http_status,
                    provider_stop_batch=provider_stop_batch,
                    batch_stop_reason=batch_stop_reason,
                )
                recorder.record(EpisodeFinished(budget=budget, snapshot=snapshot), steps)
                recorder.record(ExperimentFinished(summary=summary), steps)
                summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
            finally:
                recorder.close()
        return RunArtifacts(trace_path, summary_path, summary)
