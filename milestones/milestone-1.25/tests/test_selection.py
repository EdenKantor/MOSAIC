import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from test_calibration_protocol import PublicRulesEnvironment

from mosaic.core.config import (
    ALEM_COMMIT,
    ExperimentConfig,
    ModelConfig,
    SelectionConfig,
    load_selection_config,
)
from mosaic.core.events import ExperimentFinished, ModelCallFailed, TaskFailed, read_events
from mosaic.core.models import ModelRequest, ModelResponse, ProviderRateLimits, TokenUsage
from mosaic.experiments.run import main
from mosaic.experiments.runner import ExperimentRunner, RunArtifacts
from mosaic.experiments.selection import SelectionReport, SelectionRunner, selection_preflight
from mosaic.providers.http import FailureCategory, ProviderError


def fixture_data() -> dict[str, Any]:
    return {
        "selection_id": "offline-five-candidates",
        "stage": "selection",
        "environment": {"name": "fake", "revision": "1"},
        "agent": {"agent_id": "actor"},
        "candidates": [
            {
                "name": f"candidate-{index}",
                "role": "teacher_candidate" if index == 4 else "candidate",
                "model": {
                    "provider": "fake",
                    "model": f"fake-{index}",
                    "policy": "repair_pump" if index == 4 else "first_available",
                },
            }
            for index in range(5)
        ],
        "families": [
            {"name": "pump-a", "task": "repair_pump", "max_steps": 20},
            {"name": "pump-b", "task": "repair_pump", "max_steps": 20},
        ],
        "prompt_protocol": "minimal",
    }


def real_data() -> dict[str, Any]:
    return {
        "selection_id": "offline-provider-protocol",
        "stage": "selection",
        "environment": {"name": "alem", "revision": ALEM_COMMIT},
        "agent": {"agent_id": "actor"},
        "candidates": [
            {"name": "groq-candidate", "model": {"provider": "groq", "model": "test-groq"}},
            {"name": "gemini-candidate", "model": {"provider": "gemini", "model": "test-gemini"}},
            {
                "name": "groq-teacher-candidate",
                "role": "teacher_candidate",
                "model": {"provider": "groq", "model": "test-groq-teacher"},
            },
        ],
        "families": [
            {"name": "wood", "task": "collect_wood", "max_steps": 3},
            {"name": "stone", "task": "collect_stone", "max_steps": 3},
        ],
        "request_limits": [
            {"provider": provider, "max_requests": 100, "quota_known": True, "max_episodes": 10}
            for provider in ("gemini", "groq")
        ],
    }


def config(data: dict[str, Any]) -> SelectionConfig:
    return SelectionConfig.model_validate_json(json.dumps(data))


def public_environment(settings: ExperimentConfig) -> PublicRulesEnvironment:
    return PublicRulesEnvironment(settings.agent.agent_id)


class OfflineProvider:
    def __init__(
        self,
        settings: ModelConfig,
        calls: list[str],
        failure: FailureCategory | None = None,
        texts: tuple[str, ...] = (),
    ) -> None:
        self.settings, self.calls, self.failure, self.texts = settings, calls, failure, texts
        self.index = 0

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(self.settings.model)
        self.index += 1
        if self.failure is not None:
            raise ProviderError(
                "UNSAFE_EXCEPTION_TEXT_CANARY",
                category=self.failure,
                http_status=429 if self.failure in ("quota", "rate_limit") else None,
            )
        return ModelResponse(
            provider=self.settings.provider,
            model=request.model,
            text=self.texts[self.index - 1] if self.texts else '{"name":"Noop"}',
            usage=TokenUsage(
                input_tokens=7, measurement="provider", measurement_source="offline.fixture"
            ),
            finish_reason="stop",
        )


def test_more_than_four_independent_candidates_keep_evidence_without_role_selection(
    tmp_path: Path,
) -> None:
    settings = config(fixture_data())
    output = tmp_path / "selection"
    result = asyncio.run(SelectionRunner().run(settings, output))
    assert result.status == "completed" and result.fixture_only
    assert len(result.outcomes) == 10 and len(result.config.candidates) == 5
    assert result.selected_weak is result.selected_teacher is None
    assert result.paid_fallback_calls == 0
    assert sum(outcome.arm.metrics.task_outcome == "succeeded" for outcome in result.outcomes) == 2
    assert all(outcome.smoke_passed for outcome in result.outcomes)
    assert all(outcome.arm.initial_trace_matches for outcome in result.outcomes)
    assert len({outcome.arm.initial for outcome in result.outcomes}) == 1
    assert all(outcome.arm.summary is not None for outcome in result.outcomes)
    assert all(
        outcome.arm.summary.escalations == 0 for outcome in result.outcomes if outcome.arm.summary
    )
    assert len(result.provider_budgets) == 1
    assert result.provider_budgets[0].request_upper_bound == 200
    assert result.provider_budgets[0].budget.model_calls == 192
    assert SelectionReport.model_validate_json((output / "summary.json").read_text()) == result
    plan = json.loads((output / "batch-plan.json").read_text())
    assert plan["registered_request_upper_bounds"] == {"fake": 200}
    assert plan["automatic_retries"] == 0
    report = (output / "report.md").read_text()
    assert "candidate-4" in report and "No global model ranking" in report
    assert "real capability NOT TESTED" in report


def test_thirteen_registered_candidates_can_be_checked_offline_without_dispatching_network(
    tmp_path: Path,
) -> None:
    data = fixture_data()
    data["stage"] = "smoke"
    data["families"] = [{"name": "pump", "task": "repair_pump", "max_steps": 1}]
    template = data["candidates"][0]
    data["candidates"] = [
        {
            **template,
            "name": f"candidate-{index}",
            "model": {**template["model"], "model": f"model-{index}"},
        }
        for index in range(13)
    ]
    report = asyncio.run(SelectionRunner().run(config(data), tmp_path / "thirteen"))
    assert len(report.outcomes) == 13 and report.provider_budgets[0].budget.model_calls == 13
    registered = Path(__file__).resolve().parents[2] / "milestone-1.5" / "configs"
    real = load_selection_config(registered / "free-model-selection-20261003.yaml")
    assert len(real.candidates) == 13 and all(family.max_steps == 200 for family in real.families)
    assert real.seed == 42 and real.free_only


def test_unknown_quota_dispatches_one_episode_even_when_declared_episode_limit_is_larger(
    tmp_path: Path,
) -> None:
    data = fixture_data()
    data["request_limits"] = [{"provider": "fake", "max_requests": 200, "max_episodes": 10}]
    result = asyncio.run(SelectionRunner().run(config(data), tmp_path / "unknown-quota"))
    assert result.status == "incomplete" and len(result.outcomes) == 10
    assert result.outcomes[0].arm.budget.model_calls == 20
    assert all(outcome.arm.status == "skipped" for outcome in result.outcomes[1:])
    assert all(
        outcome.arm.reason == "provider_episode_batch_limit" for outcome in result.outcomes[1:]
    )
    assert result.provider_budgets[0].request_upper_bound == 20
    assert result.provider_budgets[0].budget.model_calls == 20


def test_partial_physical_request_cap_preserves_episode_bound_and_stops_before_reservation(
    tmp_path: Path,
) -> None:
    data = fixture_data()
    data["request_limits"] = [
        {"provider": "fake", "max_requests": 39, "quota_known": True, "max_episodes": 10}
    ]
    result = asyncio.run(SelectionRunner().run(config(data), tmp_path / "limited"))
    assert result.provider_budgets[0].request_upper_bound == 40
    assert result.provider_budgets[0].request_dispatch_limit == 39
    assert result.provider_budgets[0].budget.model_calls == 39
    partial = result.outcomes[1].arm
    assert partial.metrics.batch_stop_reason == "physical_request_batch_limit"
    assert partial.metrics.task_outcome == "unknown" and partial.metrics.provider_failures == 0
    assert partial.budget.model_calls == partial.budget.environment_actions == 19
    assert all(outcome.arm.reason == "provider_batch_stopped" for outcome in result.outcomes[2:])
    assert partial.trace_path is not None
    events = read_events(tmp_path / "limited" / partial.trace_path)
    assert not any(isinstance(event.payload, (ModelCallFailed, TaskFailed)) for event in events)


@pytest.mark.parametrize(
    "category",
    [
        "quota",
        "rate_limit",
        "billing",
        "payment",
        "insufficient_credit",
        "free_tier_unavailable",
        "server_error",
        "timeout",
        "invalid_response",
    ],
)
def test_provider_failure_pauses_shared_provider_but_preserves_other_provider_and_prior_ledger(
    tmp_path: Path, category: FailureCategory
) -> None:
    settings = config(real_data())
    calls: list[str] = []

    def models(model: ModelConfig) -> OfflineProvider:
        return OfflineProvider(model, calls, failure=category if model.provider == "groq" else None)

    output = tmp_path / "failed-provider"
    result = asyncio.run(
        SelectionRunner(public_environment, models).run(settings, output, confirm_free_tier=True)
    )
    assert result.status == "incomplete"
    assert calls.count("test-groq") == 1 and "test-groq-teacher" not in calls
    assert calls.count("test-gemini") == 6
    assert len(result.outcomes) == 6
    first = result.outcomes[0].arm
    assert first.status == "error" and first.metrics.task_outcome == "unknown"
    assert (
        first.metrics.provider_failures == 1 and first.metrics.provider_failure_category == category
    )
    assert first.budget.model_calls == 1 and first.budget.input_tokens is None
    assert first.trace_path is not None
    failures = [
        event.payload
        for event in read_events(output / first.trace_path)
        if isinstance(event.payload, ModelCallFailed)
    ]
    assert len(failures) == 1 and failures[0].provider_category == category
    assert "UNSAFE_EXCEPTION_TEXT_CANARY" not in (output / first.trace_path).read_text()
    assert all(
        outcome.arm.metrics.task_outcome == "failed"
        for outcome in result.outcomes
        if outcome.arm.provider == "gemini"
    )
    ledgers = {item.provider: item for item in result.provider_budgets}
    assert ledgers["groq"].stopped_reason == category and ledgers["groq"].budget.model_calls == 1
    assert (
        ledgers["gemini"].stopped_reason is None
        and ledgers["gemini"].budget.known_input_tokens == 42
    )
    previous_calls = len(calls)
    continued = asyncio.run(
        SelectionRunner(public_environment, models).run(
            settings,
            tmp_path / "continued",
            confirm_free_tier=True,
            candidate_names=("groq-teacher-candidate",),
            prior_reports=(result,),
        )
    )
    assert len(calls) == previous_calls
    assert all(outcome.arm.status == "skipped" for outcome in continued.outcomes)
    assert continued.prior_provider_calls == (("gemini", 6), ("groq", 1))


def test_scoped_batches_hydrate_cumulative_request_cap_before_next_reservation(
    tmp_path: Path,
) -> None:
    data = real_data()
    data["families"][0]["max_steps"] = 1
    data["request_limits"][0]["max_requests"] = 2
    settings = config(data)
    calls: list[str] = []
    runner = SelectionRunner(public_environment, lambda model: OfflineProvider(model, calls))
    first = asyncio.run(
        runner.run(
            settings,
            tmp_path / "first",
            confirm_free_tier=True,
            candidate_names=("gemini-candidate",),
            family_names=("wood",),
        )
    )
    assert first.status == "completed" and len(calls) == 1
    second = asyncio.run(
        runner.run(
            settings,
            tmp_path / "second",
            confirm_free_tier=True,
            candidate_names=("gemini-candidate",),
            family_names=("stone",),
            prior_reports=(first,),
        )
    )
    assert len(calls) == 2 and second.status == "incomplete"
    arm = second.outcomes[0].arm
    assert arm.budget.model_calls == 1 and arm.metrics.provider_failures == 0
    assert arm.metrics.batch_stop_reason == "physical_request_batch_limit"
    assert arm.metrics.task_outcome == "unknown"
    assert second.provider_budgets[0].prior_recorded_calls == 1
    assert second.provider_budgets[0].stopped_reason == "physical_request_batch_limit"


def test_prior_rate_headers_and_pacing_are_hydrated_without_real_sleep_or_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = real_data()
    data["families"] = data["families"][:1]
    data["families"][0]["max_steps"] = 1
    data["request_limits"][1].update(request_token_ceiling=8000, max_total_tokens=200000)
    settings = config(data)
    clock = [1000.0]
    waits: list[float] = []
    calls: list[str] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(time, "time", lambda: clock[0])
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    class RateProvider(OfflineProvider):
        async def generate(self, request: ModelRequest) -> ModelResponse:
            response = await super().generate(request)
            return response.model_copy(
                update={
                    "usage": TokenUsage(
                        input_tokens=7,
                        output_tokens=93,
                        provider_total_tokens=100,
                        measurement="provider",
                        measurement_source="offline.fixture",
                    ),
                    "rate_limits": ProviderRateLimits(
                        token_limit=8000,
                        remaining_requests=100,
                        remaining_tokens=0,
                        token_reset_seconds=10,
                    ),
                }
            )

    runner = SelectionRunner(public_environment, lambda model: RateProvider(model, calls))
    first = asyncio.run(
        runner.run(
            settings,
            tmp_path / "headers-first",
            confirm_free_tier=True,
            candidate_names=("groq-candidate",),
        )
    )
    second = asyncio.run(
        runner.run(
            settings,
            tmp_path / "headers-second",
            confirm_free_tier=True,
            candidate_names=("groq-teacher-candidate",),
            prior_reports=(first,),
        )
    )
    assert waits == [10] and calls == ["test-groq", "test-groq-teacher"]
    assert first.provider_budgets[0].latest_rate_limits is not None
    assert second.provider_budgets[0].prior_recorded_calls == 1
    assert second.provider_budgets[0].last_request_at == 1010


@pytest.mark.parametrize("unknown_usage", [False, True])
def test_prior_daily_token_envelope_stops_before_network_and_call_reservation(
    tmp_path: Path, unknown_usage: bool
) -> None:
    data = real_data()
    data["families"] = data["families"][:1]
    data["families"][0]["max_steps"] = 1
    data["request_limits"][1].update(request_token_ceiling=100, max_total_tokens=100)
    settings = config(data)
    calls: list[str] = []

    class TotalProvider(OfflineProvider):
        async def generate(self, request: ModelRequest) -> ModelResponse:
            response = await super().generate(request)
            return (
                response
                if unknown_usage
                else response.model_copy(
                    update={
                        "usage": TokenUsage(
                            input_tokens=7,
                            output_tokens=93,
                            provider_total_tokens=100,
                            measurement="provider",
                            measurement_source="offline.fixture",
                        )
                    }
                )
            )

    runner = SelectionRunner(public_environment, lambda model: TotalProvider(model, calls))
    first = asyncio.run(
        runner.run(
            settings,
            tmp_path / "daily-first",
            confirm_free_tier=True,
            candidate_names=("groq-candidate",),
        )
    )
    second = asyncio.run(
        runner.run(
            settings,
            tmp_path / "daily-second",
            confirm_free_tier=True,
            candidate_names=("groq-teacher-candidate",),
            prior_reports=(first,),
        )
    )
    assert calls == ["test-groq"]
    assert second.outcomes[0].arm.budget.model_calls == 0
    assert second.outcomes[0].arm.metrics.provider_failures == 0
    assert second.outcomes[0].arm.metrics.batch_stop_reason == (
        "daily_token_capacity_unknown" if unknown_usage else "daily_token_envelope_limit"
    )


def test_incomplete_trace_coverage_persists_provider_stop_across_scoped_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ExperimentRunner.run

    async def partial_trace(
        runner: ExperimentRunner, settings: ExperimentConfig, output: Path
    ) -> RunArtifacts:
        artifacts = await original(runner, settings, output)
        events = read_events(artifacts.trace_path)
        artifacts.trace_path.write_text(
            "\n".join(
                event.model_dump_json()
                for event in events
                if not isinstance(event.payload, ExperimentFinished)
            )
            + "\n",
            encoding="utf-8",
        )
        raise OSError("Offline recording interruption")

    monkeypatch.setattr(ExperimentRunner, "run", partial_trace)
    calls: list[str] = []
    runner = SelectionRunner(public_environment, lambda model: OfflineProvider(model, calls))
    settings = config(real_data())
    first = asyncio.run(
        runner.run(
            settings,
            tmp_path / "coverage-first",
            confirm_free_tier=True,
            candidate_names=("gemini-candidate",),
            family_names=("wood",),
        )
    )
    assert calls == ["test-gemini"] * 3
    assert not first.provider_budgets[0].budget_complete
    assert first.provider_budgets[0].stopped_reason == "budget_coverage_incomplete"
    second = asyncio.run(
        runner.run(
            settings,
            tmp_path / "coverage-second",
            confirm_free_tier=True,
            candidate_names=("gemini-candidate",),
            family_names=("stone",),
            prior_reports=(first,),
        )
    )
    assert calls == ["test-gemini"] * 3
    assert second.outcomes[0].arm.status == "skipped"
    assert second.outcomes[0].arm.metrics.task_outcome == "unknown"
    assert second.provider_budgets[0].stopped_reason == "budget_coverage_incomplete"


def test_unavailable_candidate_is_explicitly_skipped_without_silent_replacement(
    tmp_path: Path,
) -> None:
    data = fixture_data()
    data["candidates"][0].update(
        available=False, access_reason="model_not_available_to_current_free_tier_project"
    )
    result = asyncio.run(SelectionRunner().run(config(data), tmp_path / "unavailable"))
    excluded = [outcome for outcome in result.outcomes if outcome.candidate == "candidate-0"]
    assert len(excluded) == 2 and all(outcome.arm.status == "skipped" for outcome in excluded)
    assert all(
        outcome.arm.budget.model_calls == 0 and outcome.arm.trace_path is None
        for outcome in excluded
    )
    assert result.status == "completed" and len(result.outcomes) == 10
    assert (
        "MODEL NOT AVAILABLE TO CURRENT FREE-TIER PROJECT"
        in (tmp_path / "unavailable" / "report.md").read_text()
    )


def test_interface_failures_are_counted_separately_and_do_not_trigger_retry(tmp_path: Path) -> None:
    data = real_data()
    data["stage"] = "smoke"
    data["families"] = data["families"][:1]
    calls: list[str] = []

    def models(model: ModelConfig) -> OfflineProvider:
        return OfflineProvider(
            model, calls, texts=("not-json", '{"name":"Unknown"}', '{"name":"Noop"}')
        )

    result = asyncio.run(
        SelectionRunner(public_environment, models).run(
            config(data),
            tmp_path / "interface",
            confirm_free_tier=True,
            candidate_names=("gemini-candidate",),
        )
    )
    assert calls == ["test-gemini"] * 3
    outcome = result.outcomes[0]
    assert not outcome.smoke_passed
    assert outcome.arm.metrics.invalid_json == outcome.arm.metrics.invalid_action == 1
    assert outcome.arm.metrics.provider_failures == 0 and outcome.arm.metrics.step_cap
    assert outcome.arm.budget.model_calls == 3 and outcome.arm.budget.environment_actions == 1


def test_real_dispatch_requires_free_confirmation_and_request_limits_before_creating_output(
    tmp_path: Path,
) -> None:
    settings = config(real_data())
    with pytest.raises(ValueError, match="free-tier-only"):
        asyncio.run(SelectionRunner().run(settings, tmp_path / "unconfirmed"))
    assert not (tmp_path / "unconfirmed").exists()
    data = real_data()
    data["request_limits"] = []
    with pytest.raises(ValueError, match="request limits"):
        asyncio.run(
            SelectionRunner().run(config(data), tmp_path / "no-limits", confirm_free_tier=True)
        )
    assert not (tmp_path / "no-limits").exists()


def test_selection_preflight_only_checks_local_state_without_network_or_secret_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mosaic.experiments.selection as selection_module

    secret = "PREFLIGHT_ONLY_CREDENTIAL_CANARY"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setattr(selection_module, "_installed_alem_is_pinned", lambda: True)
    result = selection_preflight(config(real_data()))
    assert result.passed and result.inference_calls == result.network_requests == 0
    assert secret not in result.model_dump_json()


def test_cli_explicit_scope_keeps_full_registry_and_validates_unknown_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = config(fixture_data())
    source = tmp_path / "selection.yaml"
    source.write_text(settings.model_dump_json())
    output = tmp_path / "cli"
    assert (
        main(
            [
                "--selection-config",
                str(source),
                "--output",
                str(output),
                "--candidate",
                "candidate-4",
                "--family",
                "pump-b",
            ]
        )
        == 0
    )
    report = SelectionReport.model_validate_json((output / "summary.json").read_text())
    assert len(report.config.candidates) == 5 and len(report.config.families) == 2
    assert report.scope_candidates == ("candidate-4",) and report.scope_families == ("pump-b",)
    assert len(report.outcomes) == 1 and report.outcomes[0].arm.metrics.task_outcome == "succeeded"
    bad = tmp_path / "bad-scope"
    assert (
        main(
            [
                "--selection-config",
                str(source),
                "--output",
                str(bad),
                "--candidate",
                "UNSAFE_USER_NAME_CANARY",
            ]
        )
        == 2
    )
    assert not bad.exists()
    assert "UNSAFE_USER_NAME_CANARY" not in capsys.readouterr().err


def test_single_episode_cli_cannot_bypass_real_inference_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import mosaic.experiments.run as run_module

    source = tmp_path / "direct.yaml"
    source.write_text(
        json.dumps(
            {
                "experiment_id": "direct-real-provider",
                "seed": 42,
                "environment": {"name": "alem", "revision": ALEM_COMMIT},
                "agent": {"agent_id": "actor"},
                "task": "collect_wood",
                "model": {"provider": "groq", "model": "test-model"},
                "max_steps": 200,
                "prompt_protocol": "alem_official_v1",
            }
        )
    )
    called: list[str] = []
    monkeypatch.setattr(
        run_module, "provider_factory", lambda settings: called.append(settings.model)
    )
    output = tmp_path / "direct-output"
    assert main(["--config", str(source), "--output", str(output)]) == 2
    assert not called and not output.exists()
    assert "runtime setup failed" in capsys.readouterr().err


def test_model_not_found_blocks_only_that_model_without_retry_or_provider_fallback(
    tmp_path: Path,
) -> None:
    data = real_data()
    data["candidates"][0]["model"] = {"provider": "gemini", "model": "missing-model"}
    settings = config(data)
    calls: list[str] = []

    def models(model: ModelConfig) -> OfflineProvider:
        return OfflineProvider(
            model, calls, failure="not_found" if model.model == "missing-model" else None
        )

    report = asyncio.run(
        SelectionRunner(public_environment, models).run(
            settings, tmp_path / "model-missing", confirm_free_tier=True
        )
    )
    assert report.status == "incomplete"
    assert calls.count("missing-model") == 1
    assert calls.count("test-gemini") == calls.count("test-groq-teacher") == 6
    missing = [outcome.arm for outcome in report.outcomes if outcome.arm.model == "missing-model"]
    assert missing[0].metrics.provider_failure_category == "not_found"
    assert not missing[0].metrics.provider_stop_batch
    assert missing[1].status == "skipped" and missing[1].reason == "model_not_found"
    gemini = next(item for item in report.provider_budgets if item.provider == "gemini")
    assert gemini.stopped_reason is None and gemini.budget_complete
    assert gemini.unavailable_models == ("missing-model",)
    assert gemini.budget.model_calls == 7


def test_legacy_not_found_stop_allows_other_registered_model_but_cannot_retry_missing_model(
    tmp_path: Path,
) -> None:
    data = real_data()
    data["candidates"][0]["model"] = {"provider": "gemini", "model": "missing-model"}
    settings = config(data)
    calls: list[str] = []

    def models(model: ModelConfig) -> OfflineProvider:
        return OfflineProvider(
            model, calls, failure="not_found" if model.model == "missing-model" else None
        )

    runner = SelectionRunner(public_environment, models)
    first = asyncio.run(
        runner.run(
            settings,
            tmp_path / "legacy-missing",
            confirm_free_tier=True,
            candidate_names=("groq-candidate",),
            family_names=("wood",),
        )
    )
    legacy = first.model_copy(
        update={
            "provider_budgets": tuple(
                item.model_copy(
                    update={
                        "stopped_reason": "not_found",
                        "provider_stop_batch": None,
                        "unavailable_models": (),
                    }
                )
                for item in first.provider_budgets
            )
        }
    )
    second = asyncio.run(
        runner.run(
            settings,
            tmp_path / "independent-model",
            confirm_free_tier=True,
            candidate_names=("gemini-candidate",),
            family_names=("wood",),
            prior_reports=(legacy,),
        )
    )
    assert calls == ["missing-model"] + ["test-gemini"] * 3
    assert second.status == "completed" and second.provider_budgets[0].prior_recorded_calls == 1
    assert second.provider_budgets[0].stopped_reason is None
    assert second.provider_budgets[0].unavailable_models == ("missing-model",)
    third = asyncio.run(
        runner.run(
            settings,
            tmp_path / "no-retry",
            confirm_free_tier=True,
            candidate_names=("groq-candidate",),
            family_names=("stone",),
            prior_reports=(legacy, second),
        )
    )
    assert len(calls) == 4
    assert third.outcomes[0].arm.status == "skipped"
    assert third.outcomes[0].arm.reason == "model_not_found"
    assert third.provider_budgets[0].prior_recorded_calls == 4


def test_explicit_fatal_not_found_preserves_provider_stop_in_later_scopes(tmp_path: Path) -> None:
    data = real_data()
    data["candidates"][0]["model"] = {"provider": "gemini", "model": "fatal-model"}
    settings = config(data)
    calls: list[str] = []

    class FatalProvider(OfflineProvider):
        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.calls.append(self.settings.model)
            raise ProviderError("STATIC_FATAL_FAILURE", category="not_found", stop_batch=True)

    runner = SelectionRunner(public_environment, lambda model: FatalProvider(model, calls))
    first = asyncio.run(
        runner.run(
            settings,
            tmp_path / "fatal-first",
            confirm_free_tier=True,
            candidate_names=("groq-candidate",),
            family_names=("wood",),
        )
    )
    assert first.provider_budgets[0].provider_stop_batch
    second = asyncio.run(
        runner.run(
            settings,
            tmp_path / "fatal-second",
            confirm_free_tier=True,
            candidate_names=("gemini-candidate",),
            family_names=("wood",),
            prior_reports=(first,),
        )
    )
    assert calls == ["fatal-model"] and second.outcomes[0].arm.status == "skipped"
    assert second.outcomes[0].arm.reason == "provider_batch_stopped"
