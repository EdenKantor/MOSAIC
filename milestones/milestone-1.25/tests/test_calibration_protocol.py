import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mosaic.core.config import ALEM_COMMIT, CalibrationConfig, ExperimentConfig, ModelConfig
from mosaic.core.events import (
    ActionDispatched,
    ModelCalled,
    ModelCallFailed,
    ModelResponded,
    TeacherEscalated,
    TeacherEscalationEligible,
    read_events,
)
from mosaic.core.models import (
    Action,
    ActionSpec,
    EnvironmentSnapshot,
    ModelRequest,
    ModelResponse,
    Observation,
    StepResult,
    TokenUsage,
    VerificationResult,
)
from mosaic.experiments.calibration import CalibrationRunner
from mosaic.experiments.runner import ExperimentRunner
from mosaic.providers.fake import FakeModelProvider
from mosaic.providers.http import visible_messages


def real_calibration_data() -> dict[str, Any]:
    return {
        "calibration_id": "offline-config-validation",
        "environment": {"name": "alem", "revision": ALEM_COMMIT},
        "agent": {"agent_id": "calibration_agent"},
        "weak": {"provider": "ollama", "model": "local-weak-test", "max_output_tokens": 64},
        "teacher": {"provider": "groq", "model": "remote-teacher-test", "max_output_tokens": 64},
        "families": [
            {"name": "wood", "task": "collect_wood", "max_steps": 10},
            {"name": "stone", "task": "collect_stone", "max_steps": 10},
            {"name": "craft", "task": "make_wood_pickaxe", "max_steps": 10},
        ],
        "seeds": [42, 43],
        "selection_decision": "OFFLINE protocol fixture; not an empirical model selection",
        "selection_evidence_sha256": "0" * 64,
        "request_limits": [
            {"provider": provider, "max_requests": 1000, "quota_known": True, "max_episodes": 100}
            for provider in ("ollama", "groq")
        ],
    }


@pytest.mark.parametrize(
    "case",
    ["same_candidate", "duplicate_task", "temperature", "output_cap", "too_few_families"],
)
def test_real_candidate_comparison_cannot_hide_identity_or_protocol_changes(case: str) -> None:
    data = real_calibration_data()
    if case == "same_candidate":
        data["teacher"] = data["weak"].copy()
    elif case == "duplicate_task":
        data["families"][2]["task"] = "collect_wood"
    elif case == "temperature":
        data["teacher"]["temperature"] = 0.5
    elif case == "output_cap":
        data["teacher"]["max_output_tokens"] = 128
    else:
        data["families"] = data["families"][:1]
    with pytest.raises(ValidationError):
        CalibrationConfig.model_validate_json(json.dumps(data))


def test_real_protocol_defaults_use_paired_seeds_and_eight_visible_history_items() -> None:
    config = CalibrationConfig.model_validate_json(json.dumps(real_calibration_data()))
    assert config.seeds == (42, 43)
    assert config.history_limit == 8 and config.prompt_protocol == "alem_official_v1"
    assert len({family.task for family in config.families}) == 3
    assert config.weak.temperature == config.teacher.temperature
    assert config.weak.max_output_tokens == config.teacher.max_output_tokens


class PublicRulesEnvironment:
    """An offline protocol fixture; it does not emulate Alem mechanics or capability."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.steps = 0

    def reset(self, seed: int, task_id: str) -> Observation:
        self.steps = 0
        return self.observe(self.agent_id)

    def observe(self, agent_id: str) -> Observation:
        assert agent_id == self.agent_id
        return Observation(agent_id=agent_id, text=f"PUBLIC_OBSERVATION_{self.steps}")

    def available_actions(self, agent_id: str) -> tuple[ActionSpec, ...]:
        assert agent_id == self.agent_id
        return (ActionSpec(name="Noop", description="PUBLIC_ACTION_DESCRIPTION"),)

    def step(self, agent_id: str, action: Action) -> StepResult:
        assert action.name == "Noop"
        self.steps += 1
        return StepResult(
            observation=self.observe(agent_id), accepted=True, reason="PUBLIC_ACTION_FEEDBACK"
        )

    def verify(self, goal_id: str) -> VerificationResult:
        return VerificationResult(
            goal_id=goal_id, succeeded=False, evidence="PRIVATE_VERIFIER_CANARY"
        )

    def snapshot(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            state_json=json.dumps({"private": "HIDDEN_STATE_CANARY", "steps": self.steps})
        )

    def instruction_prompt(self) -> str:
        return "PUBLIC_ENVIRONMENT_RULES_CANARY"


class RecordingProvider(FakeModelProvider):
    def __init__(self) -> None:
        super().__init__("first_available")
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


def episode_config(role: str, agent_id: str) -> ExperimentConfig:
    return ExperimentConfig.model_validate_json(
        json.dumps(
            {
                "experiment_id": f"offline-{role}-protocol",
                "seed": 42,
                "environment": {"name": "alem", "revision": ALEM_COMMIT},
                "task": "collect_wood",
                "agent": {"agent_id": agent_id},
                "model": {
                    "provider": "fake",
                    "model": f"INTERNAL_{role}_MODEL_CANARY",
                    "policy": "first_available",
                },
                "max_steps": 10,
                "prompt_protocol": "alem_official_v1",
                "history_limit": 8,
                "agent_model_role": role,
            }
        )
    )


def test_independent_teacher_arm_uses_teacher_ledger_and_same_visible_prompt_as_weak(
    tmp_path: Path,
) -> None:
    actors: dict[str, RecordingProvider] = {}
    for role in ("weak", "teacher"):
        config = episode_config(role, f"INTERNAL_{role}_AGENT_CANARY")
        actor = RecordingProvider()
        actors[role] = actor
        environment = PublicRulesEnvironment(config.agent.agent_id)
        result = asyncio.run(ExperimentRunner(environment, actor).run(config, tmp_path / role))
        assert config.teacher is None and config.routing is None
        assert result.summary.status == "failed" and result.summary.reason == "max_steps"
        assert result.summary.total_model_calls == result.summary.budget.environment_actions == 10
        budget = result.summary.budget.weak if role == "weak" else result.summary.budget.teacher
        other = result.summary.budget.teacher if role == "weak" else result.summary.budget.weak
        assert budget.model_calls == 10 and other.model_calls == 0
        assert result.summary.escalations == result.summary.blocked_escalations == 0
        events = read_events(result.trace_path)
        assert not any(
            isinstance(event.payload, (TeacherEscalated, TeacherEscalationEligible))
            for event in events
        )
        calls = [event.payload for event in events if isinstance(event.payload, ModelCalled)]
        responses = [event.payload for event in events if isinstance(event.payload, ModelResponded)]
        assert len(calls) == len(responses) == len(actor.requests) == 10
        assert all(call.model_role == call.request.model_role == role for call in calls)
        assert all("skill" not in request.model_dump() for request in actor.requests)
        assert actor.requests[0].history == ()
        assert len(actor.requests[-1].history) == 8
        assert [item.step for item in actor.requests[-1].history] == list(range(2, 10))
        assert budget.input_tokens == sum(
            response.response.usage.input_tokens or 0 for response in responses
        )
        assert budget.wall_clock_ms == pytest.approx(
            sum(response.latency_ms for response in responses)
        )
        assert sum(isinstance(event.payload, ActionDispatched) for event in events) == 10
        assert "HIDDEN_STATE_CANARY" in result.trace_path.read_text()
        assert "PRIVATE_VERIFIER_CANARY" in result.trace_path.read_text()
        for request in actor.requests:
            assert "PUBLIC_ENVIRONMENT_RULES_CANARY" in request.system_prompt
            assert "PRIVATE_VERIFIER_CANARY" not in request.model_dump_json()
            assert "HIDDEN_STATE_CANARY" not in request.model_dump_json()
            prompt = json.dumps(visible_messages(request))
            assert "INTERNAL_" not in prompt
            assert "model_role" not in prompt
    assert [visible_messages(request) for request in actors["weak"].requests] == [
        visible_messages(request) for request in actors["teacher"].requests
    ]


class PartialUsageProvider:
    async def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            provider="fake",
            model=request.model,
            text='{"name":"Noop"}',
            usage=TokenUsage(
                input_tokens=7,
                reasoning_tokens=2,
                measurement="provider",
                measurement_source="offline.metadata.fixture",
            ),
            finish_reason="stop",
        )


def test_partial_usage_keeps_known_counts_and_unknown_totals_in_teacher_arm(tmp_path: Path) -> None:
    config = episode_config("teacher", "actor")
    result = asyncio.run(
        ExperimentRunner(PublicRulesEnvironment("actor"), PartialUsageProvider()).run(
            config, tmp_path / "partial"
        )
    )
    budget = result.summary.budget
    assert budget.teacher.model_calls == budget.model_calls == 10
    assert budget.weak.model_calls == 0
    assert budget.input_tokens == budget.known_input_tokens == 70
    assert budget.output_tokens is None and budget.known_output_tokens == 0
    assert budget.unknown_output_calls == budget.teacher.unknown_output_calls == 10
    assert budget.reasoning_tokens == budget.known_reasoning_tokens == 20
    assert budget.cached_input_tokens is budget.provider_total_tokens is None
    assert budget.unknown_cached_input_calls == budget.unknown_provider_total_calls == 10
    assert budget.synthetic_usage_calls == 0


class IdentityPartialUsageProvider:
    """Offline metadata stand-in; no configured model or endpoint is contacted."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    async def generate(self, request: ModelRequest) -> ModelResponse:
        response = await PartialUsageProvider().generate(request)
        return response.model_copy(update={"provider": self.config.provider})


def test_paired_preparation_matches_actual_prompts_and_aggregates_unknown_usage(
    tmp_path: Path,
) -> None:
    config = CalibrationConfig.model_validate_json(json.dumps(real_calibration_data()))
    output = tmp_path / "paired-offline-protocol"
    result = asyncio.run(
        CalibrationRunner(
            environment_factory=lambda settings: PublicRulesEnvironment(settings.agent.agent_id),
            model_factory=IdentityPartialUsageProvider,
        ).run(config, output, confirm_free_tier=True)
    )
    assert result.status == "completed" and result.excluded_pairs == 0
    assert len(result.pairs) == len(config.families) * len(config.seeds) == 6
    assert len(result.aggregates) == 2 * len(config.families)
    assert all(pair.comparable for pair in result.pairs)
    for pair in result.pairs:
        assert pair.weak.initial == pair.teacher.initial
        assert pair.weak.initial_trace_matches and pair.teacher.initial_trace_matches
        for arm in (pair.weak, pair.teacher):
            assert arm.summary is not None and arm.status == "failed"
            assert arm.summary.escalations == arm.summary.blocked_escalations == 0
            assert arm.budget.model_calls == arm.budget.environment_actions == 10
    assert all(aggregate.accepted_runs == aggregate.failed == 2 for aggregate in result.aggregates)
    assert all(aggregate.budget.input_tokens == 140 for aggregate in result.aggregates)
    assert all(aggregate.budget.output_tokens is None for aggregate in result.aggregates)
    assert result.total_budget.model_calls == result.total_budget.environment_actions == 120
    assert result.total_budget.weak.model_calls == result.total_budget.teacher.model_calls == 60
    assert result.total_budget.input_tokens == result.total_budget.known_input_tokens == 840
    assert result.total_budget.output_tokens is None
    assert result.total_budget.unknown_output_calls == 120
    assert result.total_budget.known_reasoning_tokens == result.total_budget.reasoning_tokens == 240
    assert (
        result.total_budget.provider_total_tokens is result.total_budget.cached_input_tokens is None
    )
    assert result.total_budget.unknown_provider_total_calls == 120


class FailingProvider:
    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise RuntimeError("STATIC_TEST_TRANSPORT_FAILURE")


def test_failed_independent_teacher_invocation_counts_one_unknown_call_without_retry(
    tmp_path: Path,
) -> None:
    config = episode_config("teacher", "actor")
    result = asyncio.run(
        ExperimentRunner(PublicRulesEnvironment("actor"), FailingProvider()).run(
            config, tmp_path / "failure"
        )
    )
    assert result.summary.status == "error"
    budget = result.summary.budget
    assert budget.teacher.model_calls == budget.model_calls == 1 and budget.weak.model_calls == 0
    assert budget.teacher.input_tokens is budget.teacher.output_tokens is None
    assert budget.teacher.unknown_input_calls == budget.teacher.unknown_output_calls == 1
    assert budget.environment_actions == 0
    events = read_events(result.trace_path)
    failures = [event.payload for event in events if isinstance(event.payload, ModelCallFailed)]
    assert len(failures) == 1 and failures[0].model_role == "teacher"
    assert budget.teacher.wall_clock_ms == failures[0].latency_ms


@pytest.mark.alem
@pytest.mark.skipif(
    os.environ.get("MOSAIC_TEST_ALEM") != "1", reason="Optional grounded Alem protocol check"
)
def test_grounded_solo_protocol_has_all_goals_and_reachable_crafting_with_materials() -> None:
    import jax.numpy as jnp
    from alem.alem_coop.constants import Achievement, BlockType

    from mosaic.environments.alem.calibration import AlemCalibrationEnvironment

    environment = AlemCalibrationEnvironment("actor", max_steps=10)
    assert bool(environment._env.default_params.soft_specialization)
    goals = {
        "collect_wood": Achievement.COLLECT_WOOD.value,
        "collect_stone": Achievement.COLLECT_STONE.value,
        "make_wood_pickaxe": Achievement.MAKE_WOOD_PICKAXE.value,
        "make_stone_pickaxe": Achievement.MAKE_STONE_PICKAXE.value,
    }
    for goal, index in goals.items():
        environment.reset(42, goal)
        assert not environment.verify(goal).succeeded
        state = environment._state
        environment._state = state.replace(achievements=state.achievements.at[0, index].set(True))
        assert environment.verify(goal).succeeded
    prompt = environment.instruction_prompt()
    assert "Make Wood Pickaxe" in prompt and "Make Stone Pickaxe" in prompt
    assert "1 wood" in prompt and "1 stone" in prompt and "table" in prompt.lower()

    state = environment._state
    row, column = (int(value) for value in state.player_position[0])
    level = int(state.player_level)
    neighbor_column = column + 1 if column + 1 < state.map.shape[2] else column - 1
    inventory = state.inventory.replace(
        wood=jnp.array([3], dtype=state.inventory.wood.dtype),
        stone=jnp.array([1], dtype=state.inventory.stone.dtype),
        pickaxe=jnp.array([0], dtype=state.inventory.pickaxe.dtype),
    )
    environment._state = state.replace(
        map=state.map.at[level, row, neighbor_column].set(BlockType.CRAFTING_TABLE.value),
        inventory=inventory,
    )
    actions = {action.name for action in environment.available_actions("actor")}
    assert {"Make Wood Pickaxe", "Make Stone Pickaxe"} <= actions
