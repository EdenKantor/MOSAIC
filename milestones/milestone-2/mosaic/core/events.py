"""Typed immutable event payloads; JSONL is the raw audit record."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import Field, model_validator

from mosaic.core.budget import BudgetTotals
from mosaic.core.config import ExperimentConfig, LearningConfig, TeacherAccess
from mosaic.core.models import (
    Action,
    ActionSpec,
    EnvironmentSnapshot,
    FrozenModel,
    ModelRequest,
    ModelResponse,
    ModelRole,
    Observation,
    StepResult,
    VerificationResult,
)
from mosaic.core.provenance import RunManifest
from mosaic.skills.models import CandidateSkill, SkillScope, ValidationReport, VerifiedSkill


class ExperimentStarted(FrozenModel):
    kind: Literal["ExperimentStarted"] = "ExperimentStarted"
    config: ExperimentConfig
    manifest: RunManifest


class EpisodeStarted(FrozenModel):
    kind: Literal["EpisodeStarted"] = "EpisodeStarted"
    snapshot: EnvironmentSnapshot


class ObservationReceived(FrozenModel):
    kind: Literal["ObservationReceived"] = "ObservationReceived"
    observation: Observation
    available_actions: tuple[ActionSpec, ...]


class ModelCalled(FrozenModel):
    kind: Literal["ModelCalled"] = "ModelCalled"
    call_index: int
    request: ModelRequest
    model_role: ModelRole = "weak"
    provider: str = "fake"


class ModelResponded(FrozenModel):
    kind: Literal["ModelResponded"] = "ModelResponded"
    call_index: int
    response: ModelResponse
    latency_ms: float = Field(ge=0)
    model_role: ModelRole = "weak"


class ModelCallFailed(FrozenModel):
    kind: Literal["ModelCallFailed"] = "ModelCallFailed"
    call_index: int
    model_role: ModelRole
    provider: str
    model: str
    latency_ms: float = Field(ge=0)
    error_type: str
    message: str


class WeakAttemptStarted(FrozenModel):
    kind: Literal["WeakAttemptStarted"] = "WeakAttemptStarted"
    window_steps: int


class WeakRecoveryStarted(FrozenModel):
    kind: Literal["WeakRecoveryStarted"] = "WeakRecoveryStarted"
    recovery_number: int
    window_steps: int
    reason: str


class TeacherEscalationEligible(FrozenModel):
    kind: Literal["TeacherEscalationEligible"] = "TeacherEscalationEligible"
    reason: str
    phase_steps: int
    recoveries_started: int
    consecutive_failures: int


class TeacherEscalated(FrozenModel):
    kind: Literal["TeacherEscalated"] = "TeacherEscalated"
    provider: str
    model: str
    access: TeacherAccess


class TeacherInvocationBlocked(FrozenModel):
    kind: Literal["TeacherInvocationBlocked"] = "TeacherInvocationBlocked"
    access: TeacherAccess
    reason: str


class SkillLearningStarted(FrozenModel):
    kind: Literal["SkillLearningStarted"] = "SkillLearningStarted"
    config: LearningConfig


class SkillCompilationStarted(FrozenModel):
    kind: Literal["SkillCompilationStarted"] = "SkillCompilationStarted"
    source_episode: str


class SkillCandidateCreated(FrozenModel):
    kind: Literal["SkillCandidateCreated"] = "SkillCandidateCreated"
    candidate: CandidateSkill
    compilation_ms: float = Field(ge=0)
    compilation_model_calls: Literal[0] = 0


class SkillValidationStarted(FrozenModel):
    kind: Literal["SkillValidationStarted"] = "SkillValidationStarted"
    skill_id: str
    seeds: tuple[int, ...]


class SkillValidated(FrozenModel):
    kind: Literal["SkillValidated"] = "SkillValidated"
    report: ValidationReport


class SkillRejected(FrozenModel):
    kind: Literal["SkillRejected"] = "SkillRejected"
    skill_id: str | None
    reason: str
    report: ValidationReport | None = None


class SkillPromoted(FrozenModel):
    kind: Literal["SkillPromoted"] = "SkillPromoted"
    skill: VerifiedSkill
    playbook_sha256: str
    inserted: bool


class SkillRetrievalPerformed(FrozenModel):
    kind: Literal["SkillRetrievalPerformed"] = "SkillRetrievalPerformed"
    scope: SkillScope
    observation: Observation
    playbook_sha256: str | None
    skill_id: str | None
    latency_ms: float = Field(ge=0)


class SkillExecutionStarted(FrozenModel):
    kind: Literal["SkillExecutionStarted"] = "SkillExecutionStarted"
    candidate: CandidateSkill
    mode: Literal["validation", "reuse"]


class SkillExecutionFailed(FrozenModel):
    kind: Literal["SkillExecutionFailed"] = "SkillExecutionFailed"
    skill_id: str
    completed_actions: int = Field(ge=0)
    reason: str


class SkillExecutionFinished(FrozenModel):
    kind: Literal["SkillExecutionFinished"] = "SkillExecutionFinished"
    skill_id: str
    completed_actions: int = Field(ge=0)
    environment_succeeded: bool


class SkillLearningFinished(FrozenModel):
    kind: Literal["SkillLearningFinished"] = "SkillLearningFinished"
    status: Literal["succeeded", "failed", "error"]
    reason: str
    skill_id: str | None
    budget: BudgetTotals
    compilation_calls: int = Field(ge=0)
    compilation_ms: float = Field(ge=0)


class ActionProposed(FrozenModel):
    kind: Literal["ActionProposed"] = "ActionProposed"
    action: Action
    model_role: ModelRole = "weak"


class ActionDispatched(FrozenModel):
    kind: Literal["ActionDispatched"] = "ActionDispatched"
    action: Action
    model_role: ModelRole = "weak"


class ActionExecuted(FrozenModel):
    kind: Literal["ActionExecuted"] = "ActionExecuted"
    action: Action
    result: StepResult
    model_role: ModelRole = "weak"


class ActionRejected(FrozenModel):
    kind: Literal["ActionRejected"] = "ActionRejected"
    proposal: str
    reason: str
    dispatched: bool
    result: StepResult | None = None
    model_role: ModelRole = "weak"


class EnvironmentChanged(FrozenModel):
    kind: Literal["EnvironmentChanged"] = "EnvironmentChanged"
    snapshot: EnvironmentSnapshot


class VerificationPerformed(FrozenModel):
    kind: Literal["VerificationPerformed"] = "VerificationPerformed"
    verification: VerificationResult


class TaskSucceeded(FrozenModel):
    kind: Literal["TaskSucceeded"] = "TaskSucceeded"
    verification: VerificationResult


class TaskFailed(FrozenModel):
    kind: Literal["TaskFailed"] = "TaskFailed"
    reason: str
    verification: VerificationResult | None


class RunErrored(FrozenModel):
    kind: Literal["RunErrored"] = "RunErrored"
    stage: str
    error_type: str
    message: str


class EpisodeFinished(FrozenModel):
    kind: Literal["EpisodeFinished"] = "EpisodeFinished"
    budget: BudgetTotals
    snapshot: EnvironmentSnapshot | None


class RunSummary(FrozenModel):
    experiment_id: str
    episode_id: str
    status: Literal["succeeded", "failed", "error"]
    reason: str
    steps: int
    verification: VerificationResult | None
    budget: BudgetTotals
    task: str
    total_model_calls: int = Field(ge=0)
    teacher_call_fraction: float | None = Field(ge=0, le=1)
    escalations: int = Field(ge=0)
    blocked_escalations: int = Field(ge=0)
    skill_id: str | None = None
    skill_completed: bool = False
    skill_actions: int = Field(default=0, ge=0)


class ExperimentFinished(FrozenModel):
    kind: Literal["ExperimentFinished"] = "ExperimentFinished"
    summary: RunSummary


Payload = Annotated[
    ExperimentStarted
    | EpisodeStarted
    | ObservationReceived
    | ModelCalled
    | ModelResponded
    | ModelCallFailed
    | WeakAttemptStarted
    | WeakRecoveryStarted
    | TeacherEscalationEligible
    | TeacherEscalated
    | TeacherInvocationBlocked
    | SkillLearningStarted
    | SkillCompilationStarted
    | SkillCandidateCreated
    | SkillValidationStarted
    | SkillValidated
    | SkillRejected
    | SkillPromoted
    | SkillRetrievalPerformed
    | SkillExecutionStarted
    | SkillExecutionFailed
    | SkillExecutionFinished
    | SkillLearningFinished
    | ActionProposed
    | ActionDispatched
    | ActionExecuted
    | ActionRejected
    | EnvironmentChanged
    | VerificationPerformed
    | TaskSucceeded
    | TaskFailed
    | RunErrored
    | EpisodeFinished
    | ExperimentFinished,
    Field(discriminator="kind"),
]


class Event(FrozenModel):
    schema_version: Literal[3] = 3
    event_id: str
    timestamp: datetime
    sequence: int = Field(ge=0)
    experiment_id: str
    episode_id: str
    event_type: str
    agent_id: str | None
    step: int | None
    payload: Payload

    @model_validator(mode="after")
    def consistent_type(self) -> Self:
        if self.event_type != self.payload.kind:
            raise ValueError("Event type must match its typed payload")
        return self


class EventRecorder:
    def __init__(self, path: Path, experiment_id: str, episode_id: str, agent_id: str) -> None:
        self._stream = path.open("x", encoding="utf-8", newline="\n")
        self.experiment_id = experiment_id
        self.episode_id = episode_id
        self.agent_id = agent_id
        self._sequence = 0

    def record(self, payload: Payload, step: int | None = None) -> None:
        event = Event(
            event_id=str(uuid4()),
            timestamp=datetime.now(UTC),
            sequence=self._sequence,
            experiment_id=self.experiment_id,
            episode_id=self.episode_id,
            event_type=payload.kind,
            agent_id=self.agent_id,
            step=step,
            payload=payload,
        )
        self._stream.write(event.model_dump_json() + "\n")
        self._stream.flush()
        self._sequence += 1

    def close(self) -> None:
        self._stream.close()


def read_events(path: Path) -> tuple[Event, ...]:
    return tuple(
        Event.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
    )
