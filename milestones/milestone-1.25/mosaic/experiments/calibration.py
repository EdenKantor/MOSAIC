"""Independent paired capability measurements, with no escalation or skill memory."""

import asyncio
import hashlib
import http.client
import json
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, Literal, NoReturn
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
    ProviderRequestLimit,
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
from mosaic.core.models import (
    FrozenModel,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ProviderRateLimits,
)
from mosaic.environments.base import EnvironmentAdapter
from mosaic.environments.fake import FakeEnvironment
from mosaic.experiments.runner import ExperimentRunner, InferenceBatchStopped
from mosaic.providers.base import ModelProvider
from mosaic.providers.factory import provider_factory
from mosaic.providers.http import ProviderError, visible_messages
from mosaic.providers.ollama import _local_endpoint

ROLES: tuple[ModelRole, ...] = ("weak", "teacher")


class InitialCondition(FrozenModel):
    snapshot_sha256: str
    advertised_actions_sha256: str
    visible_prompt_sha256: str


class EpisodeMetrics(FrozenModel):
    invalid_json: int = Field(default=0, ge=0)
    invalid_action: int = Field(default=0, ge=0)
    provider_failures: int = Field(default=0, ge=0)
    provider_failure_category: str | None = None
    provider_http_status: int | None = None
    provider_stop_batch: bool = False
    step_cap: bool = False
    task_outcome: Literal["succeeded", "failed", "unknown"] = "unknown"
    batch_stop_reason: str | None = None


class ProviderBudget(FrozenModel):
    provider: str
    request_upper_bound: int = Field(ge=0)
    budget: BudgetTotals
    stopped_reason: str | None = None
    budget_complete: bool = True
    request_dispatch_limit: int | None = None
    prior_recorded_calls: int = Field(default=0, ge=0)
    last_request_at: float | None = None
    latest_rate_limits: ProviderRateLimits | None = None
    rate_limits_at: float | None = None
    unavailable_models: tuple[str, ...] = ()
    provider_stop_batch: bool | None = None


class CalibrationArm(FrozenModel):
    family: str
    task: str
    seed: int
    role: ModelRole
    provider: str
    model: str
    status: Literal["succeeded", "failed", "error", "skipped"]
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
    metrics: EpisodeMetrics = Field(default_factory=EpisodeMetrics)
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
    success_rate: float | None = None
    mean_steps: float | None = None
    median_steps: float | None = None
    environment_actions: int = Field(default=0, ge=0)
    invalid_json: int = Field(default=0, ge=0)
    invalid_action: int = Field(default=0, ge=0)
    provider_failures: int = Field(default=0, ge=0)
    parse_failure_rate: float | None = None
    invalid_action_rate: float | None = None
    step_cap_rate: float | None = None
    provider_cost: None = None


class CalibrationReport(FrozenModel):
    schema_version: Literal[1] = 1
    calibration_id: str
    stage: Literal["pilot", "expanded"]
    status: Literal["completed", "incomplete", "error"]
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
    provider_budgets: tuple[ProviderBudget, ...] = ()
    paired_outcomes: tuple[tuple[str, int, str], ...] = ()
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


class BatchController:
    """Reserve episode bounds before inference and pause providers without retrying."""

    def __init__(self, limits: tuple[ProviderRequestLimit, ...]) -> None:
        self.limits: dict[str, ProviderRequestLimit] = {limit.provider: limit for limit in limits}
        self.reserved: dict[str, int] = {}
        self.episodes: dict[str, int] = {}
        self.stopped: dict[str, str] = {}
        self.last_request: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.prior_calls: dict[str, int] = {}
        self.known_total_tokens: dict[str, int] = {}
        self.unknown_total_tokens: set[str] = set()
        self.rate_limits: dict[str, ProviderRateLimits] = {}
        self.rate_limits_at: dict[str, float] = {}
        self.unavailable_models: set[tuple[str, str]] = set()

    def admit(self, provider: str, max_steps: int, model: str | None = None) -> str | None:
        if provider in self.stopped:
            return "provider_batch_stopped"
        if model is not None and (provider, model) in self.unavailable_models:
            return "model_not_found"
        limit = self.limits.get(provider)
        if limit is not None:
            episode_cap = limit.max_episodes if limit.quota_known else 1
            if self.episodes.get(provider, 0) >= episode_cap:
                return "provider_episode_batch_limit"
            if self.calls.get(provider, 0) >= limit.max_requests:
                return "provider_request_batch_limit"
        self.reserved[provider] = self.reserved.get(provider, 0) + max_steps
        self.episodes[provider] = self.episodes.get(provider, 0) + 1
        return None

    async def pace(self, provider: str) -> None:
        limit = self.limits.get(provider)
        interval = limit.min_interval_seconds if limit is not None else 0
        remaining = self.last_request.get(provider, 0) + interval - time.time()
        if remaining > 0:
            await asyncio.sleep(remaining)

    def stop(self, provider: str, reason: str) -> NoReturn:
        self.stopped[provider] = reason
        raise InferenceBatchStopped(reason)

    async def before_dispatch(self, provider: str, request: ModelRequest) -> None:
        limit = self.limits.get(provider)
        if provider in self.stopped:
            self.stop(provider, "provider_batch_stopped")
        if (provider, request.model) in self.unavailable_models:
            raise InferenceBatchStopped("model_not_found")
        if limit is not None and self.calls.get(provider, 0) >= limit.max_requests:
            self.stop(provider, "physical_request_batch_limit")
        await self.pace(provider)
        rates = self.rate_limits.get(provider)
        if rates is not None:
            if rates.remaining_requests == 0:
                self.stop(provider, "provider_request_capacity")
            for remaining, ceiling, reset, reason in (
                (
                    rates.remaining_requests,
                    1,
                    rates.request_reset_seconds,
                    "provider_request_capacity",
                ),
                (
                    rates.remaining_tokens,
                    limit.request_token_ceiling
                    if limit and limit.request_token_ceiling
                    else request.max_output_tokens,
                    rates.token_reset_seconds,
                    "provider_token_capacity",
                ),
            ):
                if remaining is not None and remaining < ceiling:
                    elapsed = time.time() - self.rate_limits_at.get(provider, time.time())
                    if reset is None:
                        self.stop(provider, reason)
                    if elapsed < reset:
                        await asyncio.sleep(reset - elapsed)
                    full = (
                        rates.request_limit
                        if reason == "provider_request_capacity"
                        else rates.token_limit
                    )
                    if full is None or full < ceiling:
                        self.stop(provider, reason)
        if limit and limit.max_total_tokens is not None:
            if provider in self.unknown_total_tokens or limit.request_token_ceiling is None:
                self.stop(provider, "daily_token_capacity_unknown")
            if (
                self.known_total_tokens.get(provider, 0) + limit.request_token_ceiling
                > limit.max_total_tokens
            ):
                self.stop(provider, "daily_token_envelope_limit")
        self.calls[provider] = self.calls.get(provider, 0) + 1
        self.last_request[provider] = time.time()

    def responded(self, provider: str, response: ModelResponse) -> None:
        total = response.usage.provider_total_tokens
        if total is None:
            self.unknown_total_tokens.add(provider)
        else:
            self.known_total_tokens[provider] = self.known_total_tokens.get(provider, 0) + total
            limit = self.limits.get(provider)
            if (
                limit
                and limit.request_token_ceiling is not None
                and total > limit.request_token_ceiling
            ):
                self.stopped[provider] = "declared_token_ceiling_exceeded"
        if response.rate_limits is not None:
            self.rate_limits[provider] = response.rate_limits
            self.rate_limits_at[provider] = time.time()

    def previous(self, item: ProviderBudget) -> None:
        self.prior_calls[item.provider] = (
            self.prior_calls.get(item.provider, 0) + item.budget.model_calls
        )
        self.calls[item.provider] = self.calls.get(item.provider, 0) + item.budget.model_calls
        self.known_total_tokens[item.provider] = (
            self.known_total_tokens.get(item.provider, 0) + item.budget.known_provider_total_tokens
        )
        if item.budget.unknown_provider_total_calls:
            self.unknown_total_tokens.add(item.provider)
        if item.stopped_reason is not None and (
            item.stopped_reason != "not_found" or item.provider_stop_batch is True
        ):
            self.stopped[item.provider] = item.stopped_reason
        self.unavailable_models.update((item.provider, model) for model in item.unavailable_models)
        if not item.budget_complete:
            self.stopped[item.provider] = "budget_coverage_incomplete"
        if item.last_request_at is not None:
            self.last_request[item.provider] = max(
                self.last_request.get(item.provider, 0), item.last_request_at
            )
        if item.latest_rate_limits is not None and (
            item.rate_limits_at or 0
        ) > self.rate_limits_at.get(item.provider, 0):
            self.rate_limits[item.provider] = item.latest_rate_limits
            self.rate_limits_at[item.provider] = item.rate_limits_at or 0

    def observe(self, arm: CalibrationArm) -> None:
        if not arm.budget_complete:
            self.stopped[arm.provider] = "budget_coverage_incomplete"
        elif (
            arm.metrics.provider_failure_category == "not_found"
            and not arm.metrics.provider_stop_batch
        ) or arm.metrics.batch_stop_reason == "model_not_found":
            self.unavailable_models.add((arm.provider, arm.model))
        elif arm.metrics.provider_failures:
            self.stopped[arm.provider] = (
                arm.metrics.provider_failure_category or "provider_inference_failure"
            )
        elif arm.metrics.batch_stop_reason:
            self.stopped[arm.provider] = arm.metrics.batch_stop_reason


class _PacedProvider:
    def __init__(self, provider: ModelProvider, identity: str, batch: BatchController) -> None:
        self.provider, self.identity, self.batch = provider, identity, batch

    async def before_dispatch(self, request: ModelRequest) -> None:
        await self.batch.before_dispatch(self.identity, request)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        response = await self.provider.generate(request)
        self.batch.responded(self.identity, response)
        return response


def _episode_metrics(
    summary: RunSummary | None, error: ProviderError | None, max_steps: int
) -> EpisodeMetrics:
    if summary is not None:
        return EpisodeMetrics(
            invalid_json=summary.invalid_json,
            invalid_action=summary.invalid_action,
            provider_failures=summary.provider_failures,
            provider_failure_category=summary.provider_failure_category,
            provider_http_status=summary.provider_http_status,
            provider_stop_batch=summary.provider_stop_batch,
            step_cap=summary.status == "failed" and summary.steps >= max_steps,
            task_outcome=summary.status if summary.status != "error" else "unknown",
            batch_stop_reason=summary.batch_stop_reason,
        )
    if error is not None:
        return EpisodeMetrics(
            provider_failures=1,
            provider_failure_category=error.category,
            provider_http_status=error.http_status,
            provider_stop_batch=error.stop_batch,
        )
    return EpisodeMetrics()


def provider_budgets(
    arms: tuple[CalibrationArm, ...], batch: BatchController
) -> tuple[ProviderBudget, ...]:
    return tuple(
        ProviderBudget(
            provider=provider,
            request_upper_bound=batch.reserved.get(provider, 0),
            budget=_merge_budgets(
                tuple(arm.budget for arm in arms if arm.provider == provider),
                sum(arm.budget.wall_clock_ms for arm in arms if arm.provider == provider),
            ),
            stopped_reason=batch.stopped.get(provider),
            budget_complete=all(arm.budget_complete for arm in arms if arm.provider == provider),
            request_dispatch_limit=batch.limits[provider].max_requests
            if provider in batch.limits
            else None,
            prior_recorded_calls=batch.prior_calls.get(provider, 0),
            last_request_at=batch.last_request.get(provider),
            latest_rate_limits=batch.rate_limits.get(provider),
            rate_limits_at=batch.rate_limits_at.get(provider),
            unavailable_models=tuple(
                sorted(
                    model for identity, model in batch.unavailable_models if identity == provider
                )
            ),
            provider_stop_batch=provider in batch.stopped,
        )
        for provider in sorted({arm.provider for arm in arms})
    )


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
    family: str,
    role: ModelRole,
    arms: tuple[CalibrationArm, ...],
    all_arms: tuple[CalibrationArm, ...] = (),
) -> CalibrationAggregate:
    values = tuple(arm.budget.weak if role == "weak" else arm.budget.teacher for arm in arms)
    calls = sum(arm.budget.model_calls for arm in arms)
    invalid_json = sum(arm.metrics.invalid_json for arm in arms)
    invalid_action = sum(arm.metrics.invalid_action for arm in arms)
    succeeded = sum(arm.metrics.task_outcome == "succeeded" for arm in arms)
    return CalibrationAggregate(
        family=family,
        role=role,
        accepted_runs=len(arms),
        succeeded=succeeded,
        failed=sum(arm.status == "failed" for arm in arms),
        errors=sum(arm.status == "error" for arm in arms),
        steps=sum(arm.steps for arm in arms),
        budget=_merge_role(values, sum(value.wall_clock_ms for value in values)),
        failure_reasons=tuple(arm.reason for arm in arms if arm.status != "succeeded"),
        success_rate=succeeded / len(arms) if arms else None,
        mean_steps=statistics.mean(arm.steps for arm in arms) if arms else None,
        median_steps=statistics.median(arm.steps for arm in arms) if arms else None,
        environment_actions=sum(arm.budget.environment_actions for arm in arms),
        invalid_json=invalid_json,
        invalid_action=invalid_action,
        provider_failures=sum(arm.metrics.provider_failures for arm in all_arms or arms),
        parse_failure_rate=invalid_json / calls if calls else None,
        invalid_action_rate=invalid_action / calls if calls else None,
        step_cap_rate=sum(arm.metrics.step_cap for arm in arms) / len(arms) if arms else None,
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
        self,
        prepared: _PreparedArm,
        family: str,
        output: Path,
        root: Path,
        blocked: str | None,
        batch: BatchController | None = None,
    ) -> CalibrationArm:
        config = prepared.config
        summary: RunSummary | None = None
        trace_path: Path | None = None
        trace_hash: str | None = None
        matches = False
        setup_ms = 0.0
        recording_error = False
        provider_error: ProviderError | None = None
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
                active: ModelProvider = (
                    _PacedProvider(provider, config.model.provider, batch) if batch else provider
                )
                result = await ExperimentRunner(prepared.environment, active).run(config, output)
                summary, trace_path = result.summary, result.trace_path
                budget = summary.budget
                budget_complete = True
                trace_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
                matches = _trace_initial_matches(trace_path, prepared.initial)
                reason = summary.reason
            except Exception as error:
                provider_error = error if isinstance(error, ProviderError) else None
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
            status=(
                summary.status
                if summary is not None and not recording_error
                else "skipped"
                if blocked is not None and not blocked.startswith("initial_")
                else "error"
            ),
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
            metrics=_episode_metrics(summary, provider_error, config.max_steps),
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "arm.json").write_text(arm.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return arm

    async def run(
        self,
        config: CalibrationConfig,
        output: Path,
        *,
        confirm_pilot_reviewed: bool = False,
        confirm_free_tier: bool = False,
    ) -> CalibrationReport:
        if config.stage == "expanded" and not confirm_pilot_reviewed:
            raise ValueError("Expanded calibration requires explicit pilot-review confirmation")
        fixture = config.weak.provider == config.teacher.provider == "fake"
        if not fixture:
            if not confirm_free_tier:
                raise ValueError("Real inference requires explicit free-tier-only confirmation")
            if not config.selection_decision or not config.selection_evidence_sha256:
                raise ValueError("Real calibration requires a recorded model selection decision")
            providers = {config.weak.provider, config.teacher.provider}
            if not providers <= {limit.provider for limit in config.request_limits}:
                raise ValueError("Real calibration requires explicit per-provider request limits")
        started = time.perf_counter()
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(
            config.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        pairs: list[CalibrationPair] = []
        batch = BatchController(config.request_limits)
        planned = {
            provider: sum(
                family.max_steps * len(config.seeds)
                for family in config.families
                for candidate in (config.weak, config.teacher)
                if candidate.provider == provider
            )
            for provider in {config.weak.provider, config.teacher.provider}
        }
        (output / "batch-plan.json").write_text(
            json.dumps(
                {
                    "registered_request_upper_bounds": planned,
                    "request_limits": [
                        limit.model_dump(mode="json") for limit in config.request_limits
                    ],
                    "unknown_quota_episode_limit_per_provider": 1,
                    "automatic_retries": 0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for family in config.families:
            if Path(family.name).name != family.name or family.name in {".", ".."}:
                raise ValueError("Unsafe calibration family path")
            for seed in config.seeds:
                weak = self._prepare(_arm_config(config, family, seed, "weak"))
                teacher = self._prepare(_arm_config(config, family, seed, "teacher"))
                mismatch = _mismatch(weak, teacher)
                weak_arm = await self._execute(
                    weak,
                    family.name,
                    output / family.name / str(seed) / "weak",
                    output,
                    mismatch
                    or batch.admit(config.weak.provider, family.max_steps, config.weak.model),
                    batch,
                )
                batch.observe(weak_arm)
                teacher_arm = await self._execute(
                    teacher,
                    family.name,
                    output / family.name / str(seed) / "teacher",
                    output,
                    mismatch
                    or batch.admit(config.teacher.provider, family.max_steps, config.teacher.model),
                    batch,
                )
                batch.observe(teacher_arm)
                comparable = (
                    mismatch is None
                    and weak_arm.initial_trace_matches
                    and teacher_arm.initial_trace_matches
                    and weak_arm.budget_complete
                    and teacher_arm.budget_complete
                    and weak_arm.metrics.task_outcome != "unknown"
                    and teacher_arm.metrics.task_outcome != "unknown"
                )
                comparison_reason = mismatch or "matched"
                if mismatch is None and not comparable:
                    comparison_reason = (
                        "incomplete_arm_budget"
                        if not weak_arm.budget_complete or not teacher_arm.budget_complete
                        else "provider_or_runtime_failure"
                        if weak_arm.metrics.provider_failures
                        or teacher_arm.metrics.provider_failures
                        else "unexecuted_arm"
                        if weak_arm.status == "skipped" or teacher_arm.status == "skipped"
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
                tuple(
                    pair.weak if role == "weak" else pair.teacher
                    for pair in pairs
                    if pair.family == family.name
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
            status="incomplete"
            if any(
                arm.metrics.provider_failures
                or arm.metrics.batch_stop_reason
                or arm.status == "skipped"
                for arm in arms
            )
            else "error"
            if any(not pair.comparable for pair in pairs)
            or any(arm.status == "error" for arm in arms)
            else "completed",
            fixture_only=fixture,
            config=config,
            config_sha256=_sha(config.model_dump(mode="json")),
            pairs=tuple(pairs),
            aggregates=aggregates,
            excluded_pairs=sum(not pair.comparable for pair in pairs),
            total_budget=total_budget,
            budget_complete=complete,
            preparation_ms=sum(arm.preparation_ms for arm in arms),
            wall_clock_ms=wall,
            provider_budgets=provider_budgets(arms, batch),
            paired_outcomes=tuple(
                (
                    pair.family,
                    pair.seed,
                    f"Weak {pair.weak.metrics.task_outcome} / "
                    f"Teacher {pair.teacher.metrics.task_outcome}"
                    if pair.comparable
                    else "EXCLUDED — infrastructure or unmatched conditions",
                )
                for pair in pairs
            ),
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
            "Provider failures are excluded from capability denominators. "
            "Registered but unexecuted arms remain visible; they are not task failures.",
            "| Family | Arm | N matched | Success rate | Mean steps | Median steps | "
            "Actions | Invalid action rate | Parse failure rate | "
            "Step-cap rate | Provider failures |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for aggregate in report.aggregates:
        rows.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    aggregate.family,
                    aggregate.role,
                    aggregate.accepted_runs,
                    aggregate.success_rate,
                    aggregate.mean_steps,
                    aggregate.median_steps,
                    aggregate.environment_actions,
                    aggregate.invalid_action_rate,
                    aggregate.parse_failure_rate,
                    aggregate.step_cap_rate,
                    aggregate.provider_failures,
                )
            )
            + " |"
        )
    rows.extend(["", "| Family | Seed | Paired outcome |", "|---|---:|---|"])
    for family_name, seed, outcome in report.paired_outcomes:
        rows.append(f"| {_cell(family_name)} | {seed} | {_cell(outcome)} |")
    rows.extend(
        [
            "",
            "Token units from different providers are reported separately and are not assumed "
            "scientifically interchangeable. No prices or paid fallback are inferred.",
            "| Provider | Reserved request upper bound | Recorded calls | Stop reason |",
            "|---|---:|---:|---|",
        ]
    )
    for item in report.provider_budgets:
        rows.append(
            f"| {_cell(item.provider)} | {item.request_upper_bound} | "
            f"{item.budget.model_calls} | {_cell(item.stopped_reason)} |"
        )
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
