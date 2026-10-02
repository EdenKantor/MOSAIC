import asyncio
import json
import random
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mosaic.core.budget import BudgetLedger
from mosaic.core.config import ExperimentConfig
from mosaic.core.events import (
    ActionDispatched,
    ActionRejected,
    EnvironmentChanged,
    EpisodeStarted,
    Event,
    ExperimentFinished,
    ModelCalled,
    ModelResponded,
    ObservationReceived,
    RunErrored,
    TaskSucceeded,
    read_events,
)
from mosaic.core.models import (
    Action,
    EnvironmentSnapshot,
    ModelRequest,
    ModelResponse,
    StepResult,
    TokenUsage,
)
from mosaic.environments.fake import FakeEnvironment
from mosaic.experiments.runner import ExperimentRunner, RunArtifacts
from mosaic.providers.base import ModelProvider
from mosaic.providers.fake import FakeModelProvider


def run(
    config: ExperimentConfig,
    path: Path,
    provider: ModelProvider | None = None,
    environment: FakeEnvironment | None = None,
) -> RunArtifacts:
    return asyncio.run(
        ExperimentRunner(
            environment or FakeEnvironment(config.agent.agent_id),
            provider or FakeModelProvider(),
        ).run(config, path)
    )


def semantic(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: semantic(item)
            for key, item in value.items()
            if key
            not in {
                "timestamp",
                "event_id",
                "episode_id",
                "latency_ms",
                "wall_clock_ms",
            }
        }
    if isinstance(value, list):
        return [semantic(item) for item in value]
    return value


def test_deterministic_trace(config: ExperimentConfig, tmp_path: Path) -> None:
    first = run(config, tmp_path / "first")
    second = run(config, tmp_path / "second")
    left, right = read_events(first.trace_path), read_events(second.trace_path)
    assert [semantic(e.model_dump(mode="json")) for e in left] == [
        semantic(e.model_dump(mode="json")) for e in right
    ]
    assert first.summary.status == second.summary.status == "succeeded"
    assert [e.sequence for e in left] == list(range(len(left)))
    assert len({e.event_id for e in left}) == len(left)
    assert all(e.timestamp.utcoffset() is not None for e in left)


def test_seed_changes_trajectory(config: ExperimentConfig, tmp_path: Path) -> None:
    other = ExperimentConfig.model_validate({**config.model_dump(), "seed": 43})
    first, second = run(config, tmp_path / "42"), run(other, tmp_path / "43")

    def snapshots(path: Path) -> list[EnvironmentSnapshot]:
        return [
            e.payload.snapshot
            for e in read_events(path)
            if isinstance(e.payload, EnvironmentChanged)
        ]

    assert snapshots(first.trace_path) != snapshots(second.trace_path)
    assert first.summary.budget.environment_actions != second.summary.budget.environment_actions


def test_seed_does_not_mutate_global_random_state() -> None:
    before = random.getstate()
    FakeEnvironment().reset(42, "repair_pump")
    assert random.getstate() == before


def test_budget_reconciles_trace(config: ExperimentConfig, tmp_path: Path) -> None:
    result = run(config, tmp_path / "run")
    events = read_events(result.trace_path)
    responses = [e.payload.response for e in events if isinstance(e.payload, ModelResponded)]
    calls = [e.payload for e in events if isinstance(e.payload, ModelCalled)]
    actions = [e.payload for e in events if isinstance(e.payload, ActionDispatched)]
    budget = result.summary.budget
    assert budget.model_calls == len(calls) == len(responses)
    assert budget.environment_actions == len(actions)
    assert budget.input_tokens == sum(r.usage.input_tokens or 0 for r in responses)
    assert budget.output_tokens == sum(r.usage.output_tokens or 0 for r in responses)
    assert budget.synthetic_usage_calls == budget.model_calls
    assert budget.unknown_input_calls == budget.unknown_output_calls == 0
    assert budget.wall_clock_ms > 0
    assert ExperimentFinished(summary=result.summary) == events[-1].payload


def test_replay_recorded_actions(config: ExperimentConfig, tmp_path: Path) -> None:
    events = read_events(run(config, tmp_path / "run").trace_path)
    environment = FakeEnvironment(config.agent.agent_id)
    environment.reset(config.seed, config.task)
    for event in events:
        if isinstance(event.payload, EpisodeStarted):
            assert environment.snapshot() == event.payload.snapshot
        elif isinstance(event.payload, ActionDispatched):
            environment.step(config.agent.agent_id, event.payload.action)
        elif isinstance(event.payload, EnvironmentChanged):
            assert environment.snapshot() == event.payload.snapshot
    assert environment.verify(config.task).succeeded


class TextProvider:
    def __init__(self, text: str) -> None:
        self.text = text

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            provider="fake",
            model=request.model,
            text=self.text,
            usage=TokenUsage(),
            finish_reason="stop",
        )


@pytest.mark.parametrize(
    "text",
    [
        "I repaired the pump. Task succeeded!",
        '{"name":"claim_success"}',
        '{"name":"wait","succeeded":true}',
        '{"name":"repair"}',
    ],
)
def test_model_cannot_claim_success(config: ExperimentConfig, tmp_path: Path, text: str) -> None:
    result = run(config, tmp_path / "run", TextProvider(text))
    events = read_events(result.trace_path)
    assert result.summary.status == "failed"
    assert result.summary.budget.model_calls == config.max_steps
    assert result.summary.budget.environment_actions == 0
    assert result.summary.budget.input_tokens is None
    assert result.summary.budget.unknown_input_calls == config.max_steps
    assert not any(isinstance(e.payload, TaskSucceeded) for e in events)
    assert sum(isinstance(e.payload, ActionRejected) for e in events) == config.max_steps


def test_max_steps_is_bounded_failure(config: ExperimentConfig, tmp_path: Path) -> None:
    result = run(config, tmp_path / "run", TextProvider('{"name":"wait"}'))
    assert result.summary.status == "failed"
    assert result.summary.reason == "max_steps"
    assert (
        result.summary.budget.model_calls
        == result.summary.budget.environment_actions
        == config.max_steps
    )


class RejectingEnvironment(FakeEnvironment):
    def step(self, agent_id: str, action: Action) -> StepResult:
        return StepResult(
            observation=self.observe(agent_id), accepted=False, reason="Rejected by ground truth"
        )


def test_environment_can_reject_advertised_action(config: ExperimentConfig, tmp_path: Path) -> None:
    result = run(config, tmp_path / "run", environment=RejectingEnvironment())
    rejected = [
        e.payload for e in read_events(result.trace_path) if isinstance(e.payload, ActionRejected)
    ]
    assert result.summary.status == "failed"
    assert len(rejected) == result.summary.budget.environment_actions == config.max_steps
    assert all(e.dispatched and e.result and not e.result.accepted for e in rejected)


class TerminatingEnvironment(FakeEnvironment):
    def step(self, agent_id: str, action: Action) -> StepResult:
        return StepResult(
            observation=self.observe(agent_id),
            accepted=True,
            reason="Episode ended without achieving goal",
            terminated=True,
        )


def test_terminal_is_not_success(config: ExperimentConfig, tmp_path: Path) -> None:
    result = run(config, tmp_path / "run", environment=TerminatingEnvironment())
    assert result.summary.status == "failed"
    assert result.summary.reason == "environment_terminated"
    assert result.summary.budget.environment_actions == 1


class FailingProvider:
    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise RuntimeError("Transport failure after dispatch")


def test_failed_call_is_accounted_and_trace_finalized(
    config: ExperimentConfig, tmp_path: Path
) -> None:
    result = run(config, tmp_path / "run", FailingProvider())
    events = read_events(result.trace_path)
    assert result.summary.status == "error"
    assert result.summary.budget.model_calls == 1
    assert result.summary.budget.environment_actions == 0
    assert result.summary.budget.input_tokens is None
    assert result.summary.budget.output_tokens is None
    assert any(isinstance(e.payload, RunErrored) for e in events)
    assert isinstance(events[-1].payload, ExperimentFinished)
    assert json.loads(result.summary_path.read_text())["status"] == "error"


def test_unknown_usage_preserves_known_subtotals() -> None:
    ledger = BudgetLedger()
    first = ledger.model_called()
    ledger.model_responded(
        first, TokenUsage(input_tokens=10, output_tokens=4, measurement="provider")
    )
    second = ledger.model_called()
    ledger.model_responded(second, TokenUsage(input_tokens=3, measurement="provider"))
    totals = ledger.totals(0)
    assert totals.input_tokens == totals.known_input_tokens == 13
    assert totals.output_tokens is None
    assert totals.known_output_tokens == 4
    assert totals.unknown_output_calls == 1


def test_config_and_event_payloads_are_immutable(config: ExperimentConfig, tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        config.model.temperature = 0.5
    event = next(
        e
        for e in read_events(run(config, tmp_path / "run").trace_path)
        if isinstance(e.payload, ObservationReceived)
    )
    assert isinstance(event.payload, ObservationReceived)
    with pytest.raises(ValidationError):
        event.payload.available_actions[0].name = "changed"
    with pytest.raises(ValidationError):
        Event.model_validate({**event.model_dump(), "event_type": "ModelCalled"})


@pytest.mark.parametrize(
    "update",
    [
        {"seed": -1},
        {"seed": True},
        {"max_steps": 0},
        {"experiment_id": "../escape"},
        {"teacher": True},
        {"task": "invented_task"},
        {"environment": {"name": "alem", "revision": "main"}},
    ],
)
def test_invalid_experimental_states(config: ExperimentConfig, update: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({**config.model_dump(), **update})


def test_existing_output_is_preserved(config: ExperimentConfig, tmp_path: Path) -> None:
    output = tmp_path / "run"
    first = run(config, output)
    original = first.trace_path.read_bytes()
    with pytest.raises(FileExistsError):
        run(config, output)
    assert first.trace_path.read_bytes() == original


def test_archive_provenance_matches_checkout_without_hashing_venv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mosaic.core import provenance

    environment, provider = FakeEnvironment(), FakeModelProvider()
    expected = provenance.manifest(environment, provider).source_sha256
    monkeypatch.setattr(provenance, "_git", lambda *args: None)
    archive = provenance.manifest(environment, provider)
    assert archive.mosaic_commit is None
    assert archive.working_tree_dirty is None
    assert archive.source_sha256 == expected


def test_usage_cannot_mislabel_known_counts_as_unavailable() -> None:
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=10, measurement="unavailable")
