import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from mosaic.core.agent import Agent
from mosaic.core.budget import BudgetLedger
from mosaic.core.config import ExperimentConfig
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
    ModelResponded,
    ObservationReceived,
    RunErrored,
    RunSummary,
    TaskFailed,
    TaskSucceeded,
    VerificationPerformed,
)
from mosaic.core.models import EnvironmentSnapshot, VerificationResult
from mosaic.core.provenance import manifest
from mosaic.environments.base import EnvironmentAdapter
from mosaic.providers.base import ModelProvider


@dataclass(frozen=True)
class RunArtifacts:
    trace_path: Path
    summary_path: Path
    summary: RunSummary


class ExperimentRunner:
    def __init__(self, environment: EnvironmentAdapter, provider: ModelProvider) -> None:
        self.environment = environment
        self.provider = provider

    async def run(self, config: ExperimentConfig, output_dir: Path) -> RunArtifacts:
        # Refuse to append to or overwrite an earlier experiment.
        output_dir.mkdir(parents=True, exist_ok=False)
        episode_id = str(uuid4())
        trace_path = output_dir / "events.jsonl"
        summary_path = output_dir / "summary.json"
        recorder = EventRecorder(
            trace_path, config.experiment_id, episode_id, config.agent.agent_id
        )
        ledger = BudgetLedger()
        agent = Agent(config)
        started = time.perf_counter()
        status: Literal["succeeded", "failed", "error"] = "failed"
        reason = "max_steps"
        stage = "provenance"
        verification: VerificationResult | None = None
        snapshot: EnvironmentSnapshot | None = None
        steps = 0
        try:
            recorder.record(
                ExperimentStarted(config=config, manifest=manifest(self.environment, self.provider))
            )
            stage = "reset"
            self.environment.reset(config.seed, config.task)
            snapshot = self.environment.snapshot()
            recorder.record(EpisodeStarted(snapshot=snapshot), step=0)
            stage = "verify"
            verification = self.environment.verify(config.task)
            if verification.goal_id != config.task:
                raise ValueError("Environment verifier returned a different goal")
            recorder.record(VerificationPerformed(verification=verification), step=0)
            if verification.succeeded:
                status, reason = "succeeded", "environment_verified"
            else:
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
                    request = agent.request(observation, actions)
                    call_index = ledger.model_called()
                    recorder.record(ModelCalled(call_index=call_index, request=request), step)
                    stage = "model_generate"
                    called = time.perf_counter()
                    response = await self.provider.generate(request)
                    latency_ms = (time.perf_counter() - called) * 1000
                    ledger.model_responded(call_index, response.usage)
                    recorder.record(
                        ModelResponded(
                            call_index=call_index, response=response, latency_ms=latency_ms
                        ),
                        step,
                    )
                    if (
                        response.provider != config.model.provider
                        or response.model != request.model
                    ):
                        raise ValueError(
                            "Provider response identity does not match immutable configuration"
                        )
                    try:
                        action = agent.parse(response.text, actions)
                    except ValueError as error:
                        recorder.record(
                            ActionRejected(
                                proposal=response.text, reason=str(error), dispatched=False
                            ),
                            step,
                        )
                        continue
                    recorder.record(ActionProposed(action=action), step)
                    stage = "environment_step"
                    ledger.action_called()
                    recorder.record(ActionDispatched(action=action), step)
                    result = self.environment.step(config.agent.agent_id, action)
                    if result.accepted:
                        recorder.record(ActionExecuted(action=action, result=result), step)
                    else:
                        recorder.record(
                            ActionRejected(
                                proposal=response.text,
                                reason=result.reason,
                                dispatched=True,
                                result=result,
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
                    if verification.succeeded:
                        status, reason = "succeeded", "environment_verified"
                        break
                    if result.terminated or result.truncated:
                        reason = (
                            "environment_terminated"
                            if result.terminated
                            else "environment_truncated"
                        )
                        break
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
                budget = ledger.totals((time.perf_counter() - started) * 1000)
                summary = RunSummary(
                    experiment_id=config.experiment_id,
                    episode_id=episode_id,
                    status=status,
                    reason=reason,
                    steps=steps,
                    verification=verification,
                    budget=budget,
                )
                recorder.record(EpisodeFinished(budget=budget, snapshot=snapshot), steps)
                recorder.record(ExperimentFinished(summary=summary), steps)
                summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
            finally:
                recorder.close()
        return RunArtifacts(trace_path, summary_path, summary)
