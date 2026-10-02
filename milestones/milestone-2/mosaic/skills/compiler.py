"""Conservative trajectory compilation, using only explicit environment-visible values."""

import hashlib
from typing import Literal

from pydantic import Field

from mosaic.core.config import CompilationConfig
from mosaic.core.events import (
    ActionDispatched,
    ActionExecuted,
    ActionRejected,
    Event,
    ExperimentFinished,
    ExperimentStarted,
    ModelCalled,
    ModelCallFailed,
    ModelResponded,
    ObservationReceived,
    TaskSucceeded,
)
from mosaic.core.models import Action, FrozenModel, Observation
from mosaic.core.provenance import configuration_sha256
from mosaic.experiments.runner import RunArtifacts
from mosaic.skills.models import (
    CandidateSkill,
    SkillProvenance,
    SkillScope,
    candidate_content_sha256,
    project_observation,
)


class VisibleTransition(FrozenModel):
    observation: Observation
    action: Action
    feedback: str
    next_observation: Observation


class VerifiedTeacherTrajectory(FrozenModel):
    verified_success: Literal[True] = True
    scope: SkillScope
    transitions: tuple[VisibleTransition, ...] = Field(min_length=1)
    provenance: SkillProvenance


def extract_teacher_trajectory(source: RunArtifacts) -> VerifiedTeacherTrajectory:
    """Audit success, then strip snapshots and verifier evidence before compilation."""
    raw = source.trace_path.read_bytes()
    events = tuple(Event.model_validate_json(line) for line in raw.decode("utf-8").splitlines())
    if not events or [event.sequence for event in events] != list(range(len(events))):
        raise ValueError("Source trace must be complete and contiguous")
    if not isinstance(events[0].payload, ExperimentStarted) or not isinstance(
        events[-1].payload, ExperimentFinished
    ):
        raise ValueError("Source trace must contain experiment boundaries")
    start, summary = events[0].payload, events[-1].payload.summary
    if summary != source.summary or any(event.episode_id != summary.episode_id for event in events):
        raise ValueError("Source artifacts disagree about their episode or outcome")
    if (
        summary.status != "succeeded"
        or summary.verification is None
        or not summary.verification.succeeded
    ):
        raise ValueError("Only environment-verified successful sources can compile")
    config = start.config
    if config.teacher is None or summary.verification.goal_id != config.task:
        raise ValueError("Source requires a teacher-assisted scoped task")
    successes = [event.payload for event in events if isinstance(event.payload, TaskSucceeded)]
    if len(successes) != 1 or successes[0].verification != summary.verification:
        raise ValueError("Source task success must match final scoped verification")
    if (
        summary.experiment_id != config.experiment_id
        or summary.task != config.task
        or any(
            event.experiment_id != config.experiment_id or event.agent_id != config.agent.agent_id
            for event in events
        )
    ):
        raise ValueError("Source trace identity does not match its configuration")
    calls: dict[int, ModelCalled] = {}
    observations: dict[int, ObservationReceived] = {}
    call_steps: dict[int, int] = {}
    responses: dict[int, Action] = {}
    dispatched: set[int] = set()
    executed: set[int] = set()
    transitions: list[VisibleTransition] = []
    teacher_started = False
    for event in events:
        payload = event.payload
        if isinstance(payload, ObservationReceived):
            if event.step is None or event.step in observations:
                raise ValueError("Source must contain one visible observation per decision")
            observations[event.step] = payload
        if isinstance(payload, (ModelCalled, ModelResponded, ActionDispatched, ActionExecuted)):
            if teacher_started and payload.model_role != "teacher":
                raise ValueError("Compiler requires uninterrupted teacher execution")
        if (
            isinstance(payload, (ActionRejected, ModelCallFailed))
            and payload.model_role == "teacher"
        ):
            raise ValueError(
                "The bounded compiler requires an uninterrupted accepted teacher procedure"
            )
        if isinstance(payload, ModelCalled) and payload.model_role == "teacher":
            teacher_started = True
            request = payload.request
            if event.step is None or event.step in calls or payload.call_index in call_steps:
                raise ValueError("Ambiguous teacher call step")
            visible = observations.get(event.step)
            if (
                visible is None
                or request.observation != visible.observation
                or request.available_actions != visible.available_actions
            ):
                raise ValueError(
                    "Teacher request must match the recorded visible observation and actions"
                )
            if (
                payload.provider != config.teacher.model.provider
                or request.model_role != "teacher"
                or request.model != config.teacher.model.model
                or request.goal != config.task
                or request.step != event.step
                or request.observation.agent_id != config.agent.agent_id
                or any(
                    feedback.observation.agent_id != config.agent.agent_id
                    for feedback in request.history
                )
            ):
                raise ValueError("Teacher request identity does not match source configuration")
            calls[event.step] = payload
            call_steps[payload.call_index] = event.step
        if isinstance(payload, ModelResponded) and payload.model_role == "teacher":
            step = call_steps.get(payload.call_index)
            if step is None or step != event.step or step in responses:
                raise ValueError("Teacher response must match one prior invocation")
            if (
                payload.response.provider != config.teacher.model.provider
                or payload.response.model != config.teacher.model.model
            ):
                raise ValueError("Teacher response identity does not match its invocation")
            action = Action.model_validate_json(payload.response.text)
            if action.name not in {spec.name for spec in calls[step].request.available_actions}:
                raise ValueError("Teacher response action must be advertised")
            responses[step] = action
        if isinstance(payload, ActionDispatched) and payload.model_role == "teacher":
            if (
                event.step is None
                or event.step in dispatched
                or responses.get(event.step) != payload.action
            ):
                raise ValueError("Teacher dispatch must match one recorded response action")
            dispatched.add(event.step)
        if isinstance(payload, ActionExecuted) and payload.model_role == "teacher":
            request_event = calls.get(event.step) if event.step is not None else None
            if (
                request_event is None
                or not payload.result.accepted
                or event.step in executed
                or event.step not in dispatched
            ):
                raise ValueError("Teacher action requires an accepted result and visible request")
            request = request_event.request
            if (
                responses.get(event.step) != payload.action
                or payload.result.observation.agent_id != config.agent.agent_id
            ):
                raise ValueError("Teacher execution must match its response and agent")
            assert event.step is not None
            executed.add(event.step)
            transitions.append(
                VisibleTransition(
                    observation=request.observation,
                    action=payload.action,
                    feedback=payload.result.reason,
                    next_observation=payload.result.observation,
                )
            )
    if (
        not transitions
        or len(transitions) != len(calls)
        or len(calls) != summary.budget.teacher.model_calls
    ):
        raise ValueError("Source must reconcile a nonempty accepted teacher trajectory")
    return VerifiedTeacherTrajectory(
        scope=SkillScope(
            environment=config.environment.name,
            revision=config.environment.revision,
            task=config.task,
        ),
        transitions=tuple(transitions),
        provenance=SkillProvenance(
            source_episode=summary.episode_id,
            source_agent=config.agent.agent_id,
            source_seed=config.seed,
            source_provider=config.teacher.model.provider,
            source_model=config.teacher.model.model,
            trace_sha256=hashlib.sha256(raw).hexdigest(),
            config_sha256=configuration_sha256(config),
            mosaic_commit=start.manifest.mosaic_commit,
        ),
    )


def compile_candidate(
    source: VerifiedTeacherTrajectory, config: CompilationConfig
) -> CandidateSkill:
    """Copy an observed bounded sequence; no inference, hidden state or claimed generalization."""
    if len(source.transitions) > config.max_actions:
        raise ValueError("Teacher trajectory exceeds the declared compilation bound")
    procedure = tuple(transition.action for transition in source.transitions)
    initiation = project_observation(
        source.transitions[0].observation.text, config.observation_fields
    )
    tools = tuple(sorted({action.name for action in procedure}))
    digest = candidate_content_sha256(
        source.scope, config.observation_fields, initiation, procedure, tools, source.scope.task
    )
    return CandidateSkill(
        skill_id=f"skill-{digest[:16]}",
        scope=source.scope,
        observation_fields=config.observation_fields,
        initiation_json=initiation,
        procedure=procedure,
        allowed_tools=tools,
        termination_goal=source.scope.task,
        provenance=source.provenance,
        content_sha256=digest,
    )
