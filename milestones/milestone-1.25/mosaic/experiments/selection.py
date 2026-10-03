"""Independent diagnostic candidates; observed names never establish empirical roles."""

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import Field

from mosaic.core.config import ExperimentConfig, ModelConfig, SelectionCandidate, SelectionConfig
from mosaic.core.events import ActionExecuted, read_events
from mosaic.core.models import FrozenModel
from mosaic.environments.base import EnvironmentAdapter
from mosaic.experiments.calibration import (
    BatchController,
    CalibrationArm,
    CalibrationRunner,
    PreflightCheck,
    ProviderBudget,
    _cell,
    _installed_alem_is_pinned,
    _mismatch,
    _sha,
    calibration_environment,
    provider_budgets,
)
from mosaic.providers.base import ModelProvider
from mosaic.providers.factory import provider_factory


class SelectionOutcome(FrozenModel):
    candidate: str
    declared_role: Literal["candidate", "teacher_candidate"]
    arm: CalibrationArm
    smoke_passed: bool | None
    trajectory_notes: tuple[str, ...]


class SelectionReport(FrozenModel):
    schema_version: Literal[1] = 1
    selection_id: str
    stage: Literal["smoke", "selection"]
    status: Literal["completed", "incomplete", "error"]
    fixture_only: bool
    config: SelectionConfig
    config_sha256: str
    scope_candidates: tuple[str, ...]
    scope_families: tuple[str, ...]
    outcomes: tuple[SelectionOutcome, ...]
    provider_budgets: tuple[ProviderBudget, ...]
    prior_provider_calls: tuple[tuple[str, int], ...] = ()
    wall_clock_ms: float = Field(ge=0)
    selected_weak: None = None
    selected_teacher: None = None
    paid_fallback_calls: Literal[0] = 0


class SelectionPreflightReport(FrozenModel):
    selection_id: str
    passed: bool
    checks: tuple[PreflightCheck, ...]
    inference_calls: Literal[0] = 0
    network_requests: Literal[0] = 0


def _access_label(candidate: SelectionCandidate) -> str:
    return {
        "model_not_available_to_current_free_tier_project": (
            "SKIPPED — MODEL NOT AVAILABLE TO CURRENT FREE-TIER PROJECT"
        ),
        "access_unverified": "SKIPPED — FREE-TIER ACCESS UNVERIFIED",
        "provider_unavailable": "SKIPPED — PROVIDER UNAVAILABLE",
    }.get(candidate.access_reason or "", "SKIPPED — ACCESS UNVERIFIED")


def selection_preflight(config: SelectionConfig) -> SelectionPreflightReport:
    """Check local prerequisites; declared model access is separate metadata evidence."""
    checks = []
    if config.environment.name == "alem":
        checks.append(
            PreflightCheck(
                name="pinned_alem",
                passed=_installed_alem_is_pinned(),
                detail="Installed integration must match the pinned source revision",
            )
        )
    for provider, variable in (("gemini", "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY")):
        if any(candidate.model.provider == provider for candidate in config.candidates):
            value = os.environ.get(variable, "")
            checks.append(
                PreflightCheck(
                    name=f"{provider}_environment_key",
                    passed=bool(value.strip()) and "\r" not in value and "\n" not in value,
                    detail="Credential availability checked without exposing its value",
                )
            )
    for candidate in config.candidates:
        checks.append(
            PreflightCheck(
                name=f"{candidate.name}_declared_access",
                passed=candidate.available,
                detail="Declared access available; retain separate metadata evidence"
                if candidate.available
                else _access_label(candidate),
            )
        )
    required = tuple(check for check in checks if not check.name.endswith("_declared_access"))
    return SelectionPreflightReport(
        selection_id=config.selection_id,
        passed=all(check.passed for check in required),
        checks=tuple(checks),
    )


def _config(
    config: SelectionConfig, candidate: SelectionCandidate, family_name: str
) -> ExperimentConfig:
    family = next(family for family in config.families if family.name == family_name)
    return ExperimentConfig(
        experiment_id=f"{config.selection_id}-{candidate.name}-{family.name}",
        seed=config.seed,
        environment=config.environment,
        task=family.task,
        agent=config.agent,
        model=candidate.model,
        max_steps=family.max_steps,
        prompt_protocol=config.prompt_protocol,
        history_limit=config.history_limit,
        agent_model_role="teacher" if candidate.role == "teacher_candidate" else "weak",
    )


def _trajectory(arm: CalibrationArm, root: Path) -> tuple[str, ...]:
    if arm.trace_path is None:
        return ("No model trajectory was dispatched.",)
    try:
        actions = tuple(
            event.payload.action.name
            for event in read_events(root / arm.trace_path)
            if isinstance(event.payload, ActionExecuted)
        )
    except (OSError, ValueError):
        return ("Trajectory record is incomplete; inspect the retained raw prefix.",)
    notes = [f"Accepted environment actions: {len(actions)}."]
    if actions:
        notes.append(f"First accepted action: {actions[0]}; last accepted action: {actions[-1]}.")
    notes.append(f"Declared task outcome: {arm.metrics.task_outcome}; termination: {arm.reason}.")
    return tuple(notes)


class SelectionRunner:
    def __init__(
        self,
        environment_factory: Callable[
            [ExperimentConfig], EnvironmentAdapter
        ] = calibration_environment,
        model_factory: Callable[[ModelConfig], ModelProvider] = provider_factory,
    ) -> None:
        self.episodes = CalibrationRunner(environment_factory, model_factory)

    async def run(
        self,
        config: SelectionConfig,
        output: Path,
        *,
        confirm_free_tier: bool = False,
        candidate_names: tuple[str, ...] | None = None,
        family_names: tuple[str, ...] | None = None,
        prior_reports: tuple[SelectionReport, ...] = (),
    ) -> SelectionReport:
        fixture = all(candidate.model.provider == "fake" for candidate in config.candidates)
        if not fixture:
            if not confirm_free_tier:
                raise ValueError("Real inference requires explicit free-tier-only confirmation")
            if config.seed != 42:
                raise ValueError("The registered diagnostic selection seed is 42")
            providers = {
                candidate.model.provider for candidate in config.candidates if candidate.available
            }
            if not providers <= {limit.provider for limit in config.request_limits}:
                raise ValueError("Real selection requires explicit per-provider request limits")
        names = candidate_names or tuple(candidate.name for candidate in config.candidates)
        families = family_names or tuple(family.name for family in config.families)
        if len(set(names)) != len(names) or not set(names) <= {
            candidate.name for candidate in config.candidates
        }:
            raise ValueError("Batch candidates must be unique registered names")
        if len(set(families)) != len(families) or not set(families) <= {
            family.name for family in config.families
        }:
            raise ValueError("Batch families must be unique registered names")
        started = time.perf_counter()
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(
            config.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        batch = BatchController(config.request_limits)
        prior_calls: dict[str, int] = {}
        for previous in prior_reports:
            for item in previous.provider_budgets:
                batch.previous(item)
                prior_calls[item.provider] = (
                    prior_calls.get(item.provider, 0) + item.budget.model_calls
                )
            for outcome in previous.outcomes:
                batch.observe(outcome.arm)
        candidates = tuple(candidate for candidate in config.candidates if candidate.name in names)
        registered_bounds = {
            provider: sum(
                family.max_steps
                for family in config.families
                for candidate in config.candidates
                if candidate.available and candidate.model.provider == provider
            )
            for provider in {candidate.model.provider for candidate in config.candidates}
        }
        scoped_bounds = {
            provider: sum(
                family.max_steps
                for family in config.families
                for candidate in candidates
                if family.name in families
                and candidate.available
                and candidate.model.provider == provider
            )
            for provider in registered_bounds
        }
        (output / "batch-plan.json").write_text(
            json.dumps(
                {
                    "registered_request_upper_bounds": registered_bounds,
                    "scope_request_upper_bounds": scoped_bounds,
                    "request_limits": [
                        limit.model_dump(mode="json") for limit in config.request_limits
                    ],
                    "prior_recorded_calls": prior_calls,
                    "prior_stopped_providers": batch.stopped,
                    "unknown_quota_episode_limit_per_provider": 1,
                    "automatic_retries": 0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        outcomes = []
        for family in config.families:
            if family.name not in families:
                continue
            prepared = {
                candidate.name: self.episodes._prepare(_config(config, candidate, family.name))
                for candidate in candidates
                if candidate.available
            }
            baseline = next(iter(prepared.values()), None)
            for candidate in candidates:
                if candidate.available:
                    current = prepared[candidate.name]
                    blocked = (
                        _mismatch(baseline, current)
                        if baseline is not None
                        else "initial_conditions_unavailable"
                    )
                    blocked = blocked or batch.admit(
                        candidate.model.provider, family.max_steps, candidate.model.model
                    )
                else:
                    current = self.episodes._prepare(_config(config, candidate, family.name))
                    blocked = candidate.access_reason or "access_unverified"
                arm = await self.episodes._execute(
                    current,
                    family.name,
                    output / candidate.name / family.name / str(config.seed),
                    output,
                    blocked,
                    batch,
                )
                batch.observe(arm)
                if arm.trace_path is None or arm.metrics.task_outcome == "unknown":
                    smoke_passed = None if arm.status == "skipped" else False
                else:
                    smoke_passed = (
                        arm.budget.environment_actions > 0
                        and arm.metrics.invalid_json
                        == arm.metrics.invalid_action
                        == arm.metrics.provider_failures
                        == 0
                        and arm.initial_trace_matches
                    )
                outcomes.append(
                    SelectionOutcome(
                        candidate=candidate.name,
                        declared_role=candidate.role,
                        arm=arm,
                        smoke_passed=smoke_passed,
                        trajectory_notes=_trajectory(arm, output),
                    )
                )
        arms = tuple(outcome.arm for outcome in outcomes)
        report = SelectionReport(
            selection_id=config.selection_id,
            stage=config.stage,
            status="incomplete"
            if any(
                arm.metrics.provider_failures
                or arm.metrics.batch_stop_reason
                or arm.reason.startswith("provider_")
                for arm in arms
            )
            else "error"
            if any(arm.status == "error" for arm in arms)
            else "completed",
            fixture_only=fixture,
            config=config,
            config_sha256=_sha(config.model_dump(mode="json")),
            scope_candidates=names,
            scope_families=families,
            outcomes=tuple(outcomes),
            provider_budgets=provider_budgets(arms, batch),
            prior_provider_calls=tuple(sorted(prior_calls.items())),
            wall_clock_ms=(time.perf_counter() - started) * 1000,
        )
        (output / "summary.json").write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (output / "report.md").write_text(render_selection_report(report), encoding="utf-8")
        return report


def render_selection_report(report: SelectionReport) -> str:
    rows = [
        f"# {report.stage.title()} {report.selection_id}",
        "",
        "ENGINEERING VERIFIED — synthetic fixture only; real capability NOT TESTED."
        if report.fixture_only
        else "EMPIRICALLY OBSERVED — diagnostic behavior on one seed; "
        "broader capability remains NOT TESTED.",
        "No global model ranking or automatic Weak/Teacher choice is made. "
        "Candidate roles remain hypotheses.",
        "Smoke task success is not required for successful infrastructure interaction.",
        f"Status: {report.status}. Configuration SHA-256: {report.config_sha256}.",
        "Provider failures and unexecuted arms are not capability failures. "
        "No retries or paid fallback.",
        "",
        "| Task | " + " | ".join(candidate.name for candidate in report.config.candidates) + " |",
        "|---|" + "---|" * len(report.config.candidates),
    ]
    for family in report.config.families:
        cells = []
        for candidate in report.config.candidates:
            outcome = next(
                (
                    item
                    for item in report.outcomes
                    if item.candidate == candidate.name and item.arm.family == family.name
                ),
                None,
            )
            cells.append(
                "NOT IN THIS BATCH"
                if outcome is None
                else _access_label(candidate)
                if not candidate.available
                else "PROVIDER FAILURE "
                f"({outcome.arm.metrics.provider_failure_category or 'unknown'})"
                if outcome.arm.metrics.provider_failures
                else f"SKIPPED ({outcome.arm.reason})"
                if outcome.arm.status == "skipped"
                else outcome.arm.metrics.task_outcome
            )
        rows.append("| " + " | ".join(_cell(value) for value in (family.task, *cells)) + " |")
    rows.extend(
        [
            "",
            "| Candidate | Task | Steps | Calls | Actions | Invalid JSON | Invalid action | "
            "Provider failures | Step cap | Smoke passes | Trajectory |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for item in report.outcomes:
        arm = item.arm
        values = (
            item.candidate,
            arm.task,
            arm.steps,
            arm.budget.model_calls,
            arm.budget.environment_actions,
            arm.metrics.invalid_json,
            arm.metrics.invalid_action,
            arm.metrics.provider_failures,
            arm.metrics.step_cap,
            item.smoke_passed,
            arm.trace_path,
        )
        rows.append("| " + " | ".join(_cell(value) for value in values) + " |")
        rows.extend(f"- {_cell(note)}" for note in item.trajectory_notes)
    rows.extend(
        [
            "",
            "Provider-reported token dimensions, known subtotals, request IDs, measurement source "
            "and latency are retained in summary.json and raw traces. Missing usage is unknown. "
            "Token units from different providers are not interchangeable.",
            "An explicit evidence-based pair-selection decision is required before the registered "
            "pilot. This diagnostic seed alone does not establish repeatability "
            "or useful transfer.",
            "Paid fallback calls: 0. Paid-tier enablement is not part of this runner.",
            "",
        ]
    )
    return "\n".join(rows)
