"""Independent paired capability measurements, with no escalation or skill memory."""

import hashlib
import http.client
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field

from mosaic.core.agent import SYSTEM_PROMPT, Agent
from mosaic.core.budget import BudgetLedger, BudgetTotals, RoleBudgetTotals
from mosaic.core.config import (
    ALEM_COMMIT,
    CalibrationConfig,
    CalibrationFamily,
    ExperimentConfig,
    ModelConfig,
)
from mosaic.core.events import (
    ActionDispatched,
    EpisodeStarted,
    Event,
    ExperimentFinished,
    ModelCalled,
    ModelCallFailed,
    ModelResponded,
    ObservationReceived,
    RunSummary,
    read_events,
)
from mosaic.core.models import FrozenModel, ModelRole
from mosaic.environments.base import EnvironmentAdapter
from mosaic.environments.fake import FakeEnvironment
from mosaic.experiments.runner import ExperimentRunner
from mosaic.providers.base import ModelProvider
from mosaic.providers.factory import provider_factory
from mosaic.providers.http import visible_messages
from mosaic.providers.ollama import _local_endpoint

ROLES: tuple[ModelRole, ...] = ("weak", "teacher")


class InitialCondition(FrozenModel):
    snapshot_sha256: str
    advertised_actions_sha256: str
    visible_prompt_sha256: str


class CalibrationArm(FrozenModel):
    family: str
    task: str
    seed: int
    role: ModelRole
    provider: str
    model: str
    status: Literal["succeeded", "failed", "error"]
    reason: str
    steps: int = Field(ge=0)
    initial: InitialCondition | None
    initial_trace_matches: bool
    budget: BudgetTotals
    budget_complete: bool
    preparation_ms: float = Field(ge=0)
    summary: RunSummary | None
    trace_path: str | None
    trace_sha256: str | None
    provider_cost: None = None


class CalibrationPair(FrozenModel):
    family: str
    seed: int
    comparable: bool
    reason: str
    weak: CalibrationArm
    teacher: CalibrationArm


class CalibrationAggregate(FrozenModel):
    family: str
    role: ModelRole
    accepted_runs: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    errors: int = Field(ge=0)
    steps: int = Field(ge=0)
    budget: RoleBudgetTotals
    failure_reasons: tuple[str, ...]
    provider_cost: None = None


class CalibrationReport(FrozenModel):
    schema_version: Literal[1] = 1
    calibration_id: str
    stage: Literal["pilot", "expanded"]
    status: Literal["completed", "error"]
    fixture_only: bool
    config: CalibrationConfig
    config_sha256: str
    pairs: tuple[CalibrationPair, ...]
    aggregates: tuple[CalibrationAggregate, ...]
    excluded_pairs: int = Field(ge=0)
    total_budget: BudgetTotals
    budget_complete: bool
    preparation_ms: float = Field(ge=0)
    wall_clock_ms: float = Field(ge=0)
    provider_cost: None = None


class PreflightCheck(FrozenModel):
    name: str
    passed: bool
    detail: str


class PreflightReport(FrozenModel):
    calibration_id: str
    passed: bool
    checks: tuple[PreflightCheck, ...]
    remote_inference_calls: Literal[0] = 0
    groq_network_requests: Literal[0] = 0


@dataclass
class _PreparedArm:
    config: ExperimentConfig
    environment: EnvironmentAdapter | None
    initial: InitialCondition | None
    preparation_ms: float
    error_reason: str | None


def _sha(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def calibration_environment(config: ExperimentConfig) -> EnvironmentAdapter:
    if config.environment.name == "fake":
        return FakeEnvironment(config.agent.agent_id)
    if config.prompt_protocol == "alem_official_v1":
        from mosaic.environments.alem.calibration import AlemCalibrationEnvironment

        return AlemCalibrationEnvironment(config.agent.agent_id, config.max_steps)
    from mosaic.environments.alem.adapter import AlemEnvironment

    return AlemEnvironment(config.agent.agent_id, config.max_steps)


def _arm_config(
    config: CalibrationConfig, family: CalibrationFamily, seed: int, role: ModelRole
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=f"{config.calibration_id}-{family.name}-{seed}-{role}",
        seed=seed,
        environment=config.environment,
        task=family.task,
        agent=config.agent,
        model=config.weak if role == "weak" else config.teacher,
        max_steps=family.max_steps,
        teacher=None,
        routing=None,
        prompt_protocol=config.prompt_protocol,
        history_limit=config.history_limit,
        agent_model_role=role,
    )


def _merge_role(values: tuple[RoleBudgetTotals, ...], wall_clock_ms: float) -> RoleBudgetTotals:
    fields: dict[str, Any] = {
        "model_calls": sum(value.model_calls for value in values),
        "synthetic_usage_calls": sum(value.synthetic_usage_calls for value in values),
        "wall_clock_ms": wall_clock_ms,
    }
    for dimension, unknown_field in (
        ("input_tokens", "unknown_input_calls"),
        ("output_tokens", "unknown_output_calls"),
        ("reasoning_tokens", "unknown_reasoning_calls"),
        ("cached_input_tokens", "unknown_cached_input_calls"),
        ("provider_total_tokens", "unknown_provider_total_calls"),
    ):
        known_field = f"known_{dimension}"
        known = sum(getattr(value, known_field) for value in values)
        unknown = sum(getattr(value, unknown_field) for value in values)
        fields[dimension] = None if unknown else known
        fields[known_field], fields[unknown_field] = known, unknown
    return RoleBudgetTotals.model_validate(fields)


def _merge_budgets(values: tuple[BudgetTotals, ...], wall_clock_ms: float) -> BudgetTotals:
    overall = _merge_role(values, wall_clock_ms)
    weak = tuple(value.weak for value in values)
    teacher = tuple(value.teacher for value in values)
    return BudgetTotals(
        **overall.model_dump(),
        environment_actions=sum(value.environment_actions for value in values),
        weak=_merge_role(weak, sum(value.wall_clock_ms for value in weak)),
        teacher=_merge_role(teacher, sum(value.wall_clock_ms for value in teacher)),
    )


def _mismatch(weak: _PreparedArm, teacher: _PreparedArm) -> str | None:
    if weak.initial is None or teacher.initial is None:
        return "initial_conditions_unavailable"
    if weak.initial.snapshot_sha256 != teacher.initial.snapshot_sha256:
        return "initial_snapshot_mismatch"
    if weak.initial.advertised_actions_sha256 != teacher.initial.advertised_actions_sha256:
        return "initial_advertised_actions_mismatch"
    if weak.initial.visible_prompt_sha256 != teacher.initial.visible_prompt_sha256:
        return "initial_visible_prompt_mismatch"
    return None


def _trace_initial_matches(path: Path, initial: InitialCondition) -> bool:
    events = read_events(path)
    snapshot = next(
        (event.payload for event in events if isinstance(event.payload, EpisodeStarted)), None
    )
    observed = next(
        (event.payload for event in events if isinstance(event.payload, ObservationReceived)), None
    )
    called = next(
        (event.payload for event in events if isinstance(event.payload, ModelCalled)), None
    )
    if (
        snapshot is None
        or _sha(snapshot.snapshot.model_dump(mode="json")) != initial.snapshot_sha256
    ):
        return False
    if (
        observed is not None
        and _sha([action.model_dump(mode="json") for action in observed.available_actions])
        != initial.advertised_actions_sha256
    ):
        return False
    return called is None or _sha(visible_messages(called.request)) == initial.visible_prompt_sha256


def _recover_trace_budget(path: Path) -> tuple[RunSummary | None, BudgetTotals, bool]:
    """Recover known reservations from a valid trace prefix after artifact failure."""
    ledger = BudgetLedger()
    reservations: dict[int, int] = {}
    final: RunSummary | None = None
    clean = True
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = Event.model_validate_json(line).payload
            if isinstance(payload, ModelCalled):
                if payload.call_index in reservations:
                    raise ValueError("Duplicate recorded reservation")
                reservations[payload.call_index] = ledger.model_called(payload.model_role)
            elif isinstance(payload, ModelResponded):
                index = reservations[payload.call_index]
                ledger.model_responded(index, payload.response.usage)
                ledger.model_finished(index, payload.latency_ms)
            elif isinstance(payload, ModelCallFailed):
                ledger.model_finished(reservations[payload.call_index], payload.latency_ms)
            elif isinstance(payload, ActionDispatched):
                ledger.action_called()
            elif isinstance(payload, ExperimentFinished):
                final = payload.summary
    except Exception:
        clean = False
    if final is not None and clean:
        return final, final.budget, True
    return final, ledger.totals(0), False


def _unknown_complete_totals(budget: BudgetTotals, roles: tuple[ModelRole, ...]) -> BudgetTotals:
    unknowns = {
        field: None
        for field in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "provider_total_tokens",
        )
    }
    changed: dict[str, Any] = dict(unknowns)
    for role in roles:
        changed[role] = getattr(budget, role).model_copy(update=unknowns)
    return budget.model_copy(update=changed)


def _aggregate(
    family: str, role: ModelRole, arms: tuple[CalibrationArm, ...]
) -> CalibrationAggregate:
    values = tuple(arm.budget.weak if role == "weak" else arm.budget.teacher for arm in arms)
    return CalibrationAggregate(
        family=family,
        role=role,
        accepted_runs=len(arms),
        succeeded=sum(arm.status == "succeeded" for arm in arms),
        failed=sum(arm.status == "failed" for arm in arms),
        errors=sum(arm.status == "error" for arm in arms),
        steps=sum(arm.steps for arm in arms),
        budget=_merge_role(values, sum(value.wall_clock_ms for value in values)),
        failure_reasons=tuple(arm.reason for arm in arms if arm.status != "succeeded"),
    )


class CalibrationRunner:
    def __init__(
        self,
        environment_factory: Callable[
            [ExperimentConfig], EnvironmentAdapter
        ] = calibration_environment,
        model_factory: Callable[[ModelConfig], ModelProvider] = provider_factory,
    ) -> None:
        self.__environment_factory = environment_factory
        self.__model_factory = model_factory

    def _prepare(self, config: ExperimentConfig) -> _PreparedArm:
        started = time.perf_counter()
        environment: EnvironmentAdapter | None = None
        initial: InitialCondition | None = None
        reason: str | None = None
        try:
            environment = self.__environment_factory(config)
            observation = environment.reset(config.seed, config.task)
            if observation.agent_id != config.agent.agent_id:
                raise ValueError("Initial observation belongs to another agent")
            actions = environment.available_actions(config.agent.agent_id)
            prompt: str | None = None
            if config.prompt_protocol == "alem_official_v1":
                prompt_builder = getattr(environment, "instruction_prompt", None)
                if not callable(prompt_builder):
                    raise ValueError("Official prompt builder is missing")
                rules = prompt_builder()
                if not isinstance(rules, str) or not rules:
                    raise ValueError("Official prompt rules must be a nonempty string")
                prompt = rules + "\n\n" + SYSTEM_PROMPT
            request = Agent(config).request(
                observation,
                actions,
                model_role=config.agent_model_role,
                step=1,
                system_prompt=prompt,
            )
            initial = InitialCondition(
                snapshot_sha256=_sha(environment.snapshot().model_dump(mode="json")),
                advertised_actions_sha256=_sha(
                    [action.model_dump(mode="json") for action in actions]
                ),
                visible_prompt_sha256=_sha(visible_messages(request)),
            )
        except Exception:
            environment, reason = None, "environment_or_prompt_setup_failed"
        return _PreparedArm(
            config, environment, initial, (time.perf_counter() - started) * 1000, reason
        )

    async def _execute(
        self, prepared: _PreparedArm, family: str, output: Path, root: Path, blocked: str | None
    ) -> CalibrationArm:
        config = prepared.config
        summary: RunSummary | None = None
        trace_path: Path | None = None
        trace_hash: str | None = None
        matches = False
        setup_ms = 0.0
        recording_error = False
        reason = prepared.error_reason or blocked
        budget = BudgetLedger().totals(0)
        budget_complete = reason is not None
        if reason is None:
            try:
                assert prepared.environment is not None and prepared.initial is not None
                constructed = time.perf_counter()
                try:
                    provider = self.__model_factory(config.model)
                finally:
                    setup_ms = (time.perf_counter() - constructed) * 1000
                result = await ExperimentRunner(prepared.environment, provider).run(config, output)
                summary, trace_path = result.summary, result.trace_path
                budget = summary.budget
                budget_complete = True
                trace_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
                matches = _trace_initial_matches(trace_path, prepared.initial)
                reason = summary.reason
            except Exception:
                recording_error = True
                reason = "arm_setup_or_recording_failed"
                possible_trace = output / "events.jsonl"
                if possible_trace.is_file():
                    summary, budget, budget_complete = _recover_trace_budget(possible_trace)
                    trace_path = possible_trace
                    try:
                        trace_hash = hashlib.sha256(possible_trace.read_bytes()).hexdigest()
                        if prepared.initial is not None:
                            matches = _trace_initial_matches(possible_trace, prepared.initial)
                    except Exception:
                        matches = False
                else:
                    budget_complete = False
        if not budget_complete:
            budget = _unknown_complete_totals(budget, (config.agent_model_role,))
        arm = CalibrationArm(
            family=family,
            task=config.task,
            seed=config.seed,
            role=config.agent_model_role,
            provider=config.model.provider,
            model=config.model.model,
            status=summary.status if summary is not None and not recording_error else "error",
            reason=reason or "arm_setup_or_recording_failed",
            steps=summary.steps if summary is not None else 0,
            initial=prepared.initial,
            initial_trace_matches=matches,
            budget=budget,
            budget_complete=budget_complete,
            preparation_ms=prepared.preparation_ms + setup_ms,
            summary=summary,
            trace_path=str(trace_path.relative_to(root)).replace("\\", "/") if trace_path else None,
            trace_sha256=trace_hash,
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "arm.json").write_text(arm.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return arm

    async def run(
        self, config: CalibrationConfig, output: Path, *, confirm_pilot_reviewed: bool = False
    ) -> CalibrationReport:
        if config.stage == "expanded" and not confirm_pilot_reviewed:
            raise ValueError("Expanded calibration requires explicit pilot-review confirmation")
        started = time.perf_counter()
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(
            config.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        pairs: list[CalibrationPair] = []
        for family in config.families:
            if Path(family.name).name != family.name or family.name in {".", ".."}:
                raise ValueError("Unsafe calibration family path")
            for seed in config.seeds:
                weak = self._prepare(_arm_config(config, family, seed, "weak"))
                teacher = self._prepare(_arm_config(config, family, seed, "teacher"))
                mismatch = _mismatch(weak, teacher)
                weak_arm = await self._execute(
                    weak, family.name, output / family.name / str(seed) / "weak", output, mismatch
                )
                teacher_arm = await self._execute(
                    teacher,
                    family.name,
                    output / family.name / str(seed) / "teacher",
                    output,
                    mismatch,
                )
                comparable = (
                    mismatch is None
                    and weak_arm.initial_trace_matches
                    and teacher_arm.initial_trace_matches
                    and weak_arm.budget_complete
                    and teacher_arm.budget_complete
                )
                comparison_reason = mismatch or "matched"
                if mismatch is None and not comparable:
                    comparison_reason = (
                        "incomplete_arm_budget"
                        if not weak_arm.budget_complete or not teacher_arm.budget_complete
                        else "recorded_initial_conditions_mismatch"
                    )
                pairs.append(
                    CalibrationPair(
                        family=family.name,
                        seed=seed,
                        comparable=comparable,
                        reason=comparison_reason,
                        weak=weak_arm,
                        teacher=teacher_arm,
                    )
                )
        aggregates = tuple(
            _aggregate(
                family.name,
                role,
                tuple(
                    pair.weak if role == "weak" else pair.teacher
                    for pair in pairs
                    if pair.family == family.name and pair.comparable
                ),
            )
            for family in config.families
            for role in ROLES
        )
        arms = tuple(arm for pair in pairs for arm in (pair.weak, pair.teacher))
        wall = (time.perf_counter() - started) * 1000
        complete = all(arm.budget_complete for arm in arms)
        total_budget = _merge_budgets(tuple(arm.budget for arm in arms), wall)
        if not complete:
            incomplete_roles = tuple(
                role
                for role in ROLES
                if any(arm.role == role and not arm.budget_complete for arm in arms)
            )
            total_budget = _unknown_complete_totals(total_budget, incomplete_roles)
        report = CalibrationReport(
            calibration_id=config.calibration_id,
            stage=config.stage,
            status="error"
            if any(not pair.comparable for pair in pairs)
            or any(arm.status == "error" for arm in arms)
            else "completed",
            fixture_only=config.weak.provider == config.teacher.provider == "fake",
            config=config,
            config_sha256=_sha(config.model_dump(mode="json")),
            pairs=tuple(pairs),
            aggregates=aggregates,
            excluded_pairs=sum(not pair.comparable for pair in pairs),
            total_budget=total_budget,
            budget_complete=complete,
            preparation_ms=sum(arm.preparation_ms for arm in arms),
            wall_clock_ms=wall,
        )
        (output / "summary.json").write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (output / "report.md").write_text(render_report(report), encoding="utf-8")
        return report


def _cell(value: object) -> str:
    return "unknown" if value is None else str(value).replace("|", "\\|").replace("\n", " ")


def _success_cell(aggregate: CalibrationAggregate) -> str:
    if not aggregate.accepted_runs:
        return "unknown"
    percent = 100 * aggregate.succeeded / aggregate.accepted_runs
    return f"{aggregate.succeeded}/{aggregate.accepted_runs} ({percent:.1f}%)"


def _cost_cell(aggregate: CalibrationAggregate) -> str:
    if not aggregate.accepted_runs:
        return "unknown (no accepted runs); monetary cost unknown"
    budget = aggregate.budget
    parts = [f"calls {budget.model_calls}"]
    for label, dimension, unknown_field in (
        ("input", "input_tokens", "unknown_input_calls"),
        ("output", "output_tokens", "unknown_output_calls"),
        ("reasoning", "reasoning_tokens", "unknown_reasoning_calls"),
    ):
        value = getattr(budget, dimension)
        detail = (
            str(value)
            if value is not None
            else (
                f"unknown (known subtotal {getattr(budget, 'known_' + dimension)}; "
                f"{getattr(budget, unknown_field)} unknown calls)"
            )
        )
        parts.append(f"{label} {detail}")
    parts.append("monetary cost unknown")
    return "; ".join(parts)


def render_report(report: CalibrationReport) -> str:
    rows = [
        f"# Calibration {report.calibration_id}",
        "",
        (
            "Deterministic fake-only fixture; these are synthetic units "
            "and do not measure real-model capability."
        )
        if report.fixture_only
        else (
            "Independent real-provider arms; provider-reported counts "
            "are retained without inferred totals or prices."
        ),
        "",
        f"Stage: {report.stage}. Status: {report.status}. "
        f"Excluded unmatched pairs: {report.excluded_pairs}.",
        "",
        "All arms contribute to total raw cost accounting, including failures and excluded pairs. "
        "Only matched pairs enter family aggregates.",
        "Provider cost: unknown. Missing token dimensions are unknown; "
        "known subtotals and unknown-call counts are in summary.json.",
        f"Call-ledger coverage complete: {report.budget_complete}. "
        "If incomplete, reported calls and subtotals are only the known recorded portion.",
        f"Configuration SHA-256: {report.config_sha256}. "
        "Source, lockfile, Python and dependency provenance are in each episode-start manifest.",
        "",
        "| Family | Weak success | Teacher success | Weak cost | Teacher cost |",
        "|---|---|---|---|---|",
    ]
    for family in report.config.families:
        weak = next(
            item for item in report.aggregates if item.family == family.name and item.role == "weak"
        )
        teacher = next(
            item
            for item in report.aggregates
            if item.family == family.name and item.role == "teacher"
        )
        rows.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    family.name,
                    _success_cell(weak),
                    _success_cell(teacher),
                    _cost_cell(weak),
                    _cost_cell(teacher),
                )
            )
            + " |"
        )
    rows.extend(
        [
            "",
            "| Family | Seed | Arm | Result | Steps | Calls | Input | Output | Reasoning | "
            "Cache | Provider total | Episode ms | Reason |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for pair in report.pairs:
        for arm in (pair.weak, pair.teacher):
            budget: RoleBudgetTotals = arm.budget
            values: tuple[object, ...] = (
                arm.family,
                arm.seed,
                arm.role,
                arm.status,
                arm.steps,
                budget.model_calls,
                budget.input_tokens,
                budget.output_tokens,
                budget.reasoning_tokens,
                budget.cached_input_tokens,
                budget.provider_total_tokens,
                round(budget.wall_clock_ms, 3),
                arm.reason,
            )
            rows.append("| " + " | ".join(_cell(value) for value in values) + " |")
    rows.extend(
        [
            "",
            "| Family | Arm | Accepted runs | Successes | Failures | Errors | Calls | Input | "
            "Output | Reasoning | Cache | Provider total | Inference ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for aggregate in report.aggregates:
        budget = aggregate.budget
        values = (
            aggregate.family,
            aggregate.role,
            aggregate.accepted_runs,
            aggregate.succeeded,
            aggregate.failed,
            aggregate.errors,
            budget.model_calls,
            budget.input_tokens,
            budget.output_tokens,
            budget.reasoning_tokens,
            budget.cached_input_tokens,
            budget.provider_total_tokens,
            round(budget.wall_clock_ms, 3),
        )
        rows.append("| " + " | ".join(_cell(value) for value in values) + " |")
    rows.extend(
        [
            "",
            f"Total calls: {report.total_budget.model_calls}; "
            f"environment actions: {report.total_budget.environment_actions}.",
            f"Total wall time: {report.wall_clock_ms:.3f} ms; "
            f"paired preparation: {report.preparation_ms:.3f} ms.",
            "Initial snapshot, advertised-action list and visible-prompt hashes "
            "are retained for both arms in summary.json. Hidden snapshots remain "
            "audit evidence and never enter model prompts.",
            "A completed calibration is a measurement result, "
            "not a decision to proceed to skill learning or withdrawal.",
            "",
        ]
    )
    return "\n".join(rows)


def _installed_alem_is_pinned() -> bool:
    try:
        direct = distribution("alem-env").read_text("direct_url.json")
        metadata = json.loads(direct) if direct else {}
        if not isinstance(metadata, dict) or not isinstance(metadata.get("vcs_info"), dict):
            return False
        return bool(metadata["vcs_info"].get("commit_id") == ALEM_COMMIT)
    except (PackageNotFoundError, OSError, ValueError):
        return False


def _ollama_models(timeout: float) -> set[str] | None:
    models: set[str] | None = None
    try:
        parsed = urlsplit(_local_endpoint())
        if parsed.hostname is None:
            return None
        connection_type = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
        try:
            connection.request("GET", "/api/tags")
            response = connection.getresponse()
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                listed = payload.get("models") if isinstance(payload, dict) else None
                if isinstance(listed, list):
                    models = {
                        name
                        for item in listed
                        if isinstance(item, dict)
                        for name in (item.get("name"), item.get("model"))
                        if isinstance(name, str)
                    }
        finally:
            connection.close()
    except Exception:
        pass
    return models


def preflight(config: CalibrationConfig) -> PreflightReport:
    checks: list[PreflightCheck] = []
    if config.environment.name == "alem":
        pinned = _installed_alem_is_pinned()
        checks.append(
            PreflightCheck(
                name="pinned_alem",
                passed=pinned,
                detail="Pinned Alem source is installed"
                if pinned
                else "Pinned Alem source is unavailable or differs",
            )
        )
    candidates = (config.weak, config.teacher)
    local = tuple(candidate for candidate in candidates if candidate.provider == "ollama")
    if local:
        models = _ollama_models(min(5.0, *(candidate.timeout_seconds for candidate in local)))
        for role, candidate in (("weak", config.weak), ("teacher", config.teacher)):
            if candidate.provider == "ollama":
                found = models is not None and candidate.model in models
                checks.append(
                    PreflightCheck(
                        name=f"{role}_ollama_model",
                        passed=found,
                        detail="Configured local model is available"
                        if found
                        else "Local service or configured model is unavailable",
                    )
                )
    if any(candidate.provider == "groq" for candidate in candidates):
        key = os.environ.get("GROQ_API_KEY", "")
        present = bool(key.strip()) and "\r" not in key and "\n" not in key
        checks.append(
            PreflightCheck(
                name="groq_environment_key",
                passed=present,
                detail="Environment credential is present; remote endpoint was not contacted"
                if present
                else "GROQ_API_KEY is missing or invalid in the environment",
            )
        )
    if all(candidate.provider == "fake" for candidate in candidates):
        checks.append(
            PreflightCheck(
                name="fixture_only",
                passed=True,
                detail="Fake-only fixture requires no model service",
            )
        )
    return PreflightReport(
        calibration_id=config.calibration_id,
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
    )
