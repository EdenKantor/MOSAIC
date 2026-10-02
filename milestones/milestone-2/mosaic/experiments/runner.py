import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from mosaic.core.agent import Agent
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
    SkillExecutionFailed,
    SkillExecutionFinished,
    SkillExecutionStarted,
    SkillRetrievalPerformed,
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
from mosaic.skills.models import CandidateSkill, SkillScope
from mosaic.skills.playbook import PlayBook
from mosaic.skills.runtime import SkillSession


@dataclass(frozen=True)
class RunArtifacts:
    trace_path: Path
    summary_path: Path
    summary: RunSummary


async def _invoke(
    provider: ModelProvider,
    request: ModelRequest,
    identity: ModelConfig,
    role: ModelRole,
    ledger: BudgetLedger,
    recorder: EventRecorder,
    step: int,
) -> ModelResponse:
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
                message=str(error),
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
        *,
        candidate: CandidateSkill | None = None,
        playbook: PlayBook | None = None,
    ) -> None:
        self.environment = environment
        self.provider = provider
        self.teacher = teacher
        self.routing_policy = routing_policy
        self.candidate = candidate
        self.playbook = playbook

    async def run(self, config: ExperimentConfig, output_dir: Path) -> RunArtifacts:
        validation_mode = bool(config.skills and config.skills.mode == "validation")
        if validation_mode:
            if self.teacher is not None or self.playbook is not None or self.candidate is None:
                raise ValueError("Validation requires only a candidate and weak provider")
            assert config.skills is not None
            if self.candidate.skill_id != config.skills.candidate_id:
                raise ValueError("Validation candidate must match immutable configuration")
            CandidateSkill.model_validate_json(self.candidate.model_dump_json())
        elif self.candidate is not None:
            raise ValueError("Unverified candidates may run only in explicit validation")
        if bool(config.skills and config.skills.mode == "reuse") != (self.playbook is not None):
            raise ValueError("Runtime PlayBook must match explicit reuse configuration")
        if self.playbook is not None:
            assert config.skills is not None and config.skills.playbook_path is not None
            if self.playbook.path.resolve() != Path(config.skills.playbook_path).resolve():
                raise ValueError("PlayBook path must match immutable configuration")
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
        role: ModelRole = "weak"
        phase: Literal["attempt", "recovery"] = "attempt"
        phase_steps = recoveries_started = consecutive_failures = 0
        history: tuple[AttemptFeedback, ...] = ()
        selected_skill = self.candidate
        skill_session: SkillSession | None = None
        book_hash = None
        try:
            book_hash = self.playbook.fingerprint() if self.playbook else None
            recorder.record(
                ExperimentStarted(
                    config=config,
                    manifest=manifest(
                        self.environment,
                        self.provider,
                        self.teacher,
                        policy,
                        playbook_sha256=book_hash,
                        candidate_content_sha256=self.candidate.content_sha256
                        if self.candidate
                        else None,
                    ),
                )
            )
            stage = "reset"
            initial_observation = self.environment.reset(config.seed, config.task)
            if initial_observation.agent_id != config.agent.agent_id:
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
                scope = SkillScope(
                    environment=config.environment.name,
                    revision=config.environment.revision,
                    task=config.task,
                )
                if self.playbook is not None:
                    stage = "skill_retrieval"
                    retrieval_started = time.perf_counter()
                    try:
                        entry = self.playbook.retrieve(scope, initial_observation)
                        if self.playbook.fingerprint() != book_hash:
                            raise ValueError(
                                "PlayBook changed during retrieval; use a single writer"
                            )
                    finally:
                        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
                        ledger.skill_retrieved(retrieval_ms)
                    selected_skill = entry.candidate if entry else None
                    recorder.record(
                        SkillRetrievalPerformed(
                            scope=scope,
                            observation=initial_observation,
                            playbook_sha256=book_hash,
                            skill_id=selected_skill.skill_id if selected_skill else None,
                            latency_ms=retrieval_ms,
                        ),
                        0,
                    )
                can_execute = True
                if selected_skill is not None:
                    if selected_skill.matches(scope, initial_observation):
                        skill_session = SkillSession(selected_skill)
                        assert config.skills is not None
                        recorder.record(
                            SkillExecutionStarted(
                                candidate=selected_skill,
                                mode=config.skills.mode,
                            ),
                            0,
                        )
                    elif validation_mode:
                        can_execute = False
                        reason = "skill_initiation_not_matched"
                        ledger.skill_failed()
                        recorder.record(
                            SkillExecutionFailed(
                                skill_id=selected_skill.skill_id,
                                completed_actions=0,
                                reason=reason,
                            ),
                            0,
                        )
                if config.routing:
                    recorder.record(
                        WeakAttemptStarted(window_steps=config.routing.attempt_steps), 0
                    )
                for step in range(1, config.max_steps + 1) if can_execute else ():
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
                    skill_hint = skill_session.hint() if skill_session and role == "weak" else None
                    if skill_hint is not None:
                        ledger.skill_prompted(len(skill_hint.model_dump_json().encode("utf-8")))
                    request = agent.request(
                        observation,
                        actions,
                        model=identity,
                        model_role=role,
                        history=history if config.routing else (),
                        step=step,
                        skill=skill_hint,
                    )
                    stage = "model_generate"
                    response = await _invoke(
                        active_provider, request, identity, role, ledger, recorder, step
                    )
                    action = None
                    environment_done = False
                    feedback_observation = observation
                    try:
                        action = agent.parse(response.text, actions)
                    except ValueError as error:
                        accepted, feedback = False, str(error)
                        recorder.record(
                            ActionRejected(
                                proposal=response.text,
                                reason=feedback,
                                dispatched=False,
                                model_role=role,
                            ),
                            step,
                        )
                    else:
                        recorder.record(ActionProposed(action=action, model_role=role), step)
                        stage = "environment_step"
                        ledger.action_called()
                        recorder.record(ActionDispatched(action=action, model_role=role), step)
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
                                ActionExecuted(action=action, result=result, model_role=role), step
                            )
                        else:
                            recorder.record(
                                ActionRejected(
                                    proposal=response.text,
                                    reason=result.reason,
                                    dispatched=True,
                                    result=result,
                                    model_role=role,
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
                    if skill_hint is not None:
                        assert skill_session is not None and selected_skill is not None
                        if not skill_session.advance(action, accepted):
                            ledger.skill_failed()
                            recorder.record(
                                SkillExecutionFailed(
                                    skill_id=selected_skill.skill_id,
                                    completed_actions=skill_session.cursor,
                                    reason="weak_action_rejected_or_diverged",
                                ),
                                step,
                            )
                            if validation_mode:
                                reason = "skill_execution_failed"
                                break
                        elif skill_session.completed:
                            recorder.record(
                                SkillExecutionFinished(
                                    skill_id=selected_skill.skill_id,
                                    completed_actions=skill_session.cursor,
                                    environment_succeeded=verification.succeeded,
                                ),
                                step,
                            )
                            if not verification.succeeded:
                                skill_session.failed = True
                                ledger.skill_failed()
                                recorder.record(
                                    SkillExecutionFailed(
                                        skill_id=selected_skill.skill_id,
                                        completed_actions=skill_session.cursor,
                                        reason="skill_termination_unverified",
                                    ),
                                    step,
                                )
                                if validation_mode:
                                    reason = "skill_termination_unverified"
                                    break
                    if config.routing:
                        history = (
                            *history,
                            AttemptFeedback(
                                step=step,
                                model_role=role,
                                proposal=response.text,
                                action=action,
                                accepted=accepted,
                                feedback=feedback,
                                observation=feedback_observation,
                            ),
                        )[-config.routing.history_limit :]
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
            status, reason = "error", f"{stage}: {type(error).__name__}: {error}"
            recorder.record(
                RunErrored(stage=stage, error_type=type(error).__name__, message=str(error)), steps
            )
            recorder.record(TaskFailed(reason=reason, verification=verification), steps)
        finally:
            try:
                if (
                    skill_session is not None
                    and not skill_session.failed
                    and not skill_session.completed
                ):
                    ledger.skill_failed()
                    skill_session.failed = True
                    recorder.record(
                        SkillExecutionFailed(
                            skill_id=skill_session.candidate.skill_id,
                            completed_actions=skill_session.cursor,
                            reason="run_ended_before_skill_completion",
                        ),
                        steps,
                    )
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
                    skill_id=selected_skill.skill_id if selected_skill else None,
                    skill_completed=skill_session.completed if skill_session else False,
                    skill_actions=skill_session.cursor if skill_session else 0,
                )
                recorder.record(EpisodeFinished(budget=budget, snapshot=snapshot), steps)
                recorder.record(ExperimentFinished(summary=summary), steps)
                summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
            finally:
                recorder.close()
        return RunArtifacts(trace_path, summary_path, summary)
