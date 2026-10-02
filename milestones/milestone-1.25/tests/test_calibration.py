import asyncio
import hashlib
import http.client
import json
from pathlib import Path
from typing import Any

import pytest
from test_calibration_protocol import real_calibration_data

import mosaic.experiments.calibration as calibration_module
from mosaic.core.config import (
    CalibrationConfig,
    ExperimentConfig,
    ModelConfig,
    load_calibration_config,
)
from mosaic.core.events import (
    ExperimentFinished,
    ExperimentStarted,
    ModelCalled,
    ModelResponded,
    read_events,
)
from mosaic.core.models import EnvironmentSnapshot
from mosaic.environments.fake import FakeEnvironment
from mosaic.experiments.calibration import CalibrationReport, CalibrationRunner, preflight
from mosaic.experiments.run import main
from mosaic.experiments.runner import ExperimentRunner, RunArtifacts
from mosaic.providers.factory import provider_factory

PROJECT = Path(__file__).resolve().parents[1]


def fixture_data() -> dict[str, Any]:
    return {
        "calibration_id": "offline-paired-calibration",
        "environment": {"name": "fake", "revision": "1"},
        "agent": {"agent_id": "actor"},
        "weak": {"provider": "fake", "model": "fake-weak", "policy": "first_available"},
        "teacher": {"provider": "fake", "model": "fake-teacher", "policy": "repair_pump"},
        "families": [{"name": "pump", "task": "repair_pump", "max_steps": 20}],
        "seeds": [42, 43],
        "prompt_protocol": "minimal",
    }


@pytest.fixture
def fixture_config() -> CalibrationConfig:
    return CalibrationConfig.model_validate_json(json.dumps(fixture_data()))


def test_paired_fixed_schedule_preserves_every_artifact_and_role_cost(
    fixture_config: CalibrationConfig, tmp_path: Path
) -> None:
    output = tmp_path / "calibration"
    result = asyncio.run(CalibrationRunner().run(fixture_config, output))
    persisted = CalibrationReport.model_validate_json((output / "summary.json").read_text())
    assert persisted == result
    canonical_config = json.dumps(
        fixture_config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert result.config_sha256 == hashlib.sha256(canonical_config.encode()).hexdigest()
    assert result.budget_complete and all(
        arm.budget_complete for pair in result.pairs for arm in (pair.weak, pair.teacher)
    )
    assert result.status == "completed" and result.fixture_only and result.excluded_pairs == 0
    assert [(pair.family, pair.seed) for pair in result.pairs] == [("pump", 42), ("pump", 43)]
    assert all(pair.comparable and pair.reason == "matched" for pair in result.pairs)
    assert result.total_budget.model_calls == result.total_budget.environment_actions == 62
    assert (
        result.total_budget.weak.model_calls == 40 and result.total_budget.teacher.model_calls == 22
    )
    assert result.provider_cost is None
    recorded_calls = []
    recorded_responses = []
    for pair in result.pairs:
        assert pair.weak.initial == pair.teacher.initial
        assert pair.weak.status == "failed" and pair.teacher.status == "succeeded"
        for arm in (pair.weak, pair.teacher):
            assert arm.summary is not None and arm.initial_trace_matches
            assert arm.trace_path is not None and arm.trace_sha256 is not None
            path = output / arm.trace_path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == arm.trace_sha256
            assert json.loads((path.parent / "arm.json").read_text()) == arm.model_dump(mode="json")
            events = read_events(path)
            started = next(
                event.payload for event in events if isinstance(event.payload, ExperimentStarted)
            )
            assert started.config.seed == pair.seed
            assert started.config.teacher is None and started.config.routing is None
            assert started.config.max_steps == 20 and started.config.prompt_protocol == "minimal"
            assert started.config.agent_model_role == arm.role
            calls = [event.payload for event in events if isinstance(event.payload, ModelCalled)]
            responses = [
                event.payload for event in events if isinstance(event.payload, ModelResponded)
            ]
            assert arm.budget.model_calls == len(calls) == len(responses)
            assert all(call.model_role == call.request.model_role == arm.role for call in calls)
            recorded_calls.extend(calls)
            recorded_responses.extend(responses)
    assert result.total_budget.model_calls == len(recorded_calls)
    assert result.total_budget.input_tokens == sum(
        response.response.usage.input_tokens or 0 for response in recorded_responses
    )
    assert result.total_budget.output_tokens == sum(
        response.response.usage.output_tokens or 0 for response in recorded_responses
    )
    assert result.total_budget.teacher.wall_clock_ms == pytest.approx(
        sum(
            response.latency_ms
            for response in recorded_responses
            if response.model_role == "teacher"
        )
    )
    aggregates = {aggregate.role: aggregate for aggregate in result.aggregates}
    assert aggregates["weak"].accepted_runs == aggregates["weak"].failed == 2
    assert aggregates["teacher"].accepted_runs == aggregates["teacher"].succeeded == 2
    assert aggregates["weak"].budget.model_calls == 40
    assert aggregates["teacher"].budget.model_calls == 22
    report = (output / "report.md").read_text()
    assert "synthetic" in report and "unknown" in report
    assert "| Family | Weak success | Teacher success | Weak cost | Teacher cost |" in report
    assert "0/2 (0.0%)" in report and "2/2 (100.0%)" in report
    assert result.config_sha256 in report


class SnapshotMismatchEnvironment(FakeEnvironment):
    def __init__(self, agent_id: str, identity: str) -> None:
        super().__init__(agent_id)
        self.identity = identity

    def snapshot(self) -> EnvironmentSnapshot:
        original = json.loads(super().snapshot().state_json)
        return EnvironmentSnapshot(
            state_json=json.dumps({**original, "audit_identity": self.identity})
        )


def test_initial_snapshot_mismatch_blocks_both_model_factories_and_keeps_pair(
    fixture_config: CalibrationConfig, tmp_path: Path
) -> None:
    factories_called: list[str] = []

    def environment(settings: ExperimentConfig) -> FakeEnvironment:
        return SnapshotMismatchEnvironment(settings.agent.agent_id, settings.agent_model_role)

    def model(settings: ModelConfig) -> Any:
        factories_called.append(settings.model)
        raise AssertionError("A mismatched pair must not create either provider")

    result = asyncio.run(
        CalibrationRunner(environment, model).run(fixture_config, tmp_path / "mismatch")
    )
    assert not factories_called
    assert result.status == "error" and result.excluded_pairs == 2
    assert len(result.pairs) == 2
    assert all(
        not pair.comparable and pair.reason == "initial_snapshot_mismatch" for pair in result.pairs
    )
    assert all(pair.weak.status == pair.teacher.status == "error" for pair in result.pairs)
    assert result.total_budget.model_calls == result.total_budget.environment_actions == 0
    assert all(aggregate.accepted_runs == 0 for aggregate in result.aggregates)


def test_one_provider_setup_error_preserves_later_registered_pair_and_earlier_costs(
    fixture_config: CalibrationConfig, tmp_path: Path
) -> None:
    count = 0

    def model(settings: ModelConfig) -> Any:
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError("Offline provider initialization failed")
        return provider_factory(settings)

    result = asyncio.run(
        CalibrationRunner(model_factory=model).run(fixture_config, tmp_path / "partial")
    )
    assert count == 4 and len(result.pairs) == 2
    assert result.status == "error" and result.pairs[0].weak.status == "error"
    assert result.pairs[0].teacher.summary is not None
    assert result.pairs[0].teacher.budget.teacher.model_calls == 16
    assert result.pairs[1].comparable
    assert result.pairs[1].weak.budget.weak.model_calls == 20
    assert result.pairs[1].teacher.budget.teacher.model_calls == 6
    assert result.total_budget.weak.model_calls == 20
    assert result.total_budget.teacher.model_calls == 22
    assert result.total_budget.model_calls == 42


@pytest.mark.parametrize("invalid_tail", [False, True])
def test_incomplete_recording_retains_known_costs_and_excludes_only_affected_pair(
    fixture_config: CalibrationConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_tail: bool,
) -> None:
    original = ExperimentRunner.run
    attempted: list[tuple[int, str]] = []
    known_input = 0
    known_output = 0

    async def recording_failure(
        runner: ExperimentRunner, config: ExperimentConfig, output: Path
    ) -> RunArtifacts:
        nonlocal known_input, known_output
        attempted.append((config.seed, config.agent_model_role))
        artifacts = await original(runner, config, output)
        if len(attempted) == 1:
            events = read_events(artifacts.trace_path)
            responses = [
                event.payload for event in events if isinstance(event.payload, ModelResponded)
            ]
            known_input = sum(response.response.usage.input_tokens or 0 for response in responses)
            known_output = sum(response.response.usage.output_tokens or 0 for response in responses)
            prefix = [
                event.model_dump_json()
                for event in events
                if not isinstance(event.payload, ExperimentFinished)
            ]
            if invalid_tail:
                prefix.append('{"interrupted_event":')
            artifacts.trace_path.write_text("\n".join(prefix) + "\n", encoding="utf-8")
            raise OSError("Offline interruption during artifact recording")
        return artifacts

    monkeypatch.setattr(ExperimentRunner, "run", recording_failure)
    output = tmp_path / "interrupted"
    result = asyncio.run(CalibrationRunner().run(fixture_config, output))
    assert attempted == [(42, "weak"), (42, "teacher"), (43, "weak"), (43, "teacher")]
    assert result.status == "error" and not result.budget_complete
    assert result.excluded_pairs == 1 and result.pairs[1].comparable
    pair = result.pairs[0]
    assert not pair.comparable and pair.reason == "incomplete_arm_budget"
    assert pair.weak.status == "error" and pair.weak.summary is None
    assert not pair.weak.budget_complete and pair.teacher.budget_complete
    assert pair.weak.budget.model_calls == pair.weak.budget.weak.model_calls == 20
    assert pair.weak.budget.environment_actions == 20
    assert pair.weak.budget.known_input_tokens == known_input
    assert pair.weak.budget.known_output_tokens == known_output
    assert pair.weak.budget.teacher.model_calls == 0
    for dimension in (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "provider_total_tokens",
    ):
        assert getattr(pair.weak.budget, dimension) is None
        assert getattr(pair.weak.budget.weak, dimension) is None
        assert getattr(result.total_budget, dimension) is None
        assert getattr(result.total_budget.weak, dimension) is None
    assert result.total_budget.model_calls == result.total_budget.environment_actions == 62
    assert result.total_budget.teacher.model_calls == 22
    assert result.total_budget.teacher.input_tokens is not None
    assert result.total_budget.known_input_tokens > known_input
    aggregates = {aggregate.role: aggregate for aggregate in result.aggregates}
    assert aggregates["weak"].accepted_runs == aggregates["teacher"].accepted_runs == 1
    assert aggregates["weak"].budget.model_calls == 20
    assert aggregates["teacher"].budget.model_calls == 6
    assert "Call-ledger coverage complete: False" in (output / "report.md").read_text()


def test_terminal_trace_retains_complete_ledger_after_later_artifact_write_failure(
    fixture_config: CalibrationConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ExperimentRunner.run
    attempted = 0

    async def later_failure(
        runner: ExperimentRunner, config: ExperimentConfig, output: Path
    ) -> RunArtifacts:
        nonlocal attempted
        attempted += 1
        artifacts = await original(runner, config, output)
        if attempted == 1:
            assert isinstance(read_events(artifacts.trace_path)[-1].payload, ExperimentFinished)
            raise OSError("Offline summary write failure after completed trace")
        return artifacts

    monkeypatch.setattr(ExperimentRunner, "run", later_failure)
    result = asyncio.run(CalibrationRunner().run(fixture_config, tmp_path / "terminal"))
    assert attempted == 4 and result.status == "error"
    assert result.budget_complete and result.excluded_pairs == 0
    assert all(pair.comparable for pair in result.pairs)
    first = result.pairs[0].weak
    assert first.status == "error" and first.reason == "arm_setup_or_recording_failed"
    assert first.summary is not None and first.summary.status == "failed"
    assert first.budget_complete and first.initial_trace_matches
    assert first.budget == first.summary.budget
    assert result.total_budget.model_calls == result.total_budget.environment_actions == 62
    assert result.total_budget.input_tokens is not None
    aggregates = {aggregate.role: aggregate for aggregate in result.aggregates}
    assert aggregates["weak"].accepted_runs == aggregates["teacher"].accepted_runs == 2
    assert aggregates["weak"].errors == aggregates["weak"].failed == 1
    assert aggregates["teacher"].succeeded == 2


def test_expanded_schedule_requires_review_before_creating_outputs_or_providers(
    tmp_path: Path,
) -> None:
    data = fixture_data()
    data["stage"] = "expanded"
    config = CalibrationConfig.model_validate_json(json.dumps(data))
    output = tmp_path / "expanded"
    with pytest.raises(ValueError, match="pilot-review"):
        asyncio.run(CalibrationRunner().run(config, output))
    assert not output.exists()
    result = asyncio.run(CalibrationRunner().run(config, output, confirm_pilot_reviewed=True))
    assert result.stage == "expanded" and result.status == "completed"


def test_existing_calibration_evidence_is_not_overwritten(
    fixture_config: CalibrationConfig, tmp_path: Path
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_bytes(b"Existing evidence\n")
    with pytest.raises(FileExistsError):
        asyncio.run(CalibrationRunner().run(fixture_config, output))
    assert sentinel.read_bytes() == b"Existing evidence\n"


@pytest.mark.parametrize("has_key", [False, True])
def test_preflight_reads_only_local_inventory_and_never_contacts_groq(
    monkeypatch: pytest.MonkeyPatch, has_key: bool
) -> None:
    config = CalibrationConfig.model_validate_json(json.dumps(real_calibration_data()))
    secret = "GROQ_PREFLIGHT_SECRET_CANARY"
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setenv("GROQ_API_KEY", secret if has_key else "")
    monkeypatch.setattr(calibration_module, "_installed_alem_is_pinned", lambda: True)
    requests: list[tuple[str, str]] = []

    class Response:
        status = 200

        def read(self) -> bytes:
            return json.dumps({"models": [{"name": config.weak.model}]}).encode()

    class Connection:
        def __init__(self, host: str, port: int, timeout: float) -> None:
            assert host == "127.0.0.1" and port == 11434 and timeout <= 5

        def request(self, method: str, path: str) -> None:
            requests.append((method, path))

        def getresponse(self) -> Response:
            return Response()

        def close(self) -> None:
            pass

    def forbidden_remote(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Preflight must not create a remote HTTPS connection")

    monkeypatch.setattr(http.client, "HTTPConnection", Connection)
    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden_remote)
    report = preflight(config)
    assert report.passed is has_key
    assert report.remote_inference_calls == report.groq_network_requests == 0
    assert requests == [("GET", "/api/tags")]
    assert secret not in report.model_dump_json()


@pytest.mark.parametrize("source", ["--config", "--calibration-config"])
def test_cli_rejects_credentials_in_config_without_echoing_secret(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str
) -> None:
    data = (
        fixture_data()
        if source == "--calibration-config"
        else {
            "experiment_id": "bad-secret-config",
            "seed": 42,
            "environment": {"name": "fake", "revision": "1"},
            "task": "repair_pump",
            "agent": {"agent_id": "actor"},
            "model": {"provider": "fake", "model": "fake"},
            "max_steps": 2,
        }
    )
    secret = "FORBIDDEN_CONFIG_SECRET_CANARY"
    data["api_key"] = secret
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(data))
    output = tmp_path / "bad-output"
    assert main([source, str(path), "--output-dir", str(output)]) == 2
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "Configuration validation failed" in captured.err
    assert not output.exists()


def test_cli_runs_only_fixture_by_default_and_supports_output_alias(
    fixture_config: CalibrationConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(fixture_config.model_dump_json())
    output = tmp_path / "cli"
    assert main(["--calibration-config", str(config_path), "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert "Fake-only synthetic fixture" in captured.out
    assert "62" in captured.out
    assert CalibrationReport.model_validate_json((output / "summary.json").read_text()).fixture_only


def test_default_real_pilot_and_review_gated_expansion_share_protocol_and_task_registry() -> None:
    configs = PROJECT.parent / "milestone-1.5" / "configs"
    pilot = load_calibration_config(configs / "calibration-pilot.yaml")
    expanded = load_calibration_config(configs / "calibration-expanded.yaml")
    assert pilot.stage == "pilot" and len(pilot.seeds) == 5
    assert expanded.stage == "expanded" and len(expanded.seeds) == 20
    assert set(pilot.seeds) <= set(expanded.seeds)
    assert pilot.families == expanded.families
    assert {family.task for family in pilot.families} == {
        "collect_wood",
        "collect_stone",
        "make_wood_pickaxe",
        "make_stone_pickaxe",
    }
    assert pilot.weak == expanded.weak and pilot.teacher == expanded.teacher
    assert pilot.prompt_protocol == expanded.prompt_protocol == "alem_official_v1"
    assert pilot.history_limit == expanded.history_limit == 8
    assert pilot.weak.max_output_tokens == pilot.teacher.max_output_tokens
