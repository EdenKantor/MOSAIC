import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from mosaic.core.budget import BudgetLedger, RoleBudgetTotals
from mosaic.core.models import ActionSpec, ModelRequest, Observation, TokenUsage
from mosaic.experiments.learning import merge_budgets
from mosaic.providers.fake import FakeModelProvider
from mosaic.skills.models import ValidationTrial

TOKEN_DIMENSIONS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "provider_total_tokens",
)


def test_absent_provider_dimensions_remain_unknown() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=20, measurement="provider")
    assert usage.input_tokens == 100 and usage.output_tokens == 20
    assert (
        usage.reasoning_tokens is usage.cached_input_tokens is usage.provider_total_tokens is None
    )
    assert usage.provider_request_id is usage.measurement_source is None
    assert TokenUsage.model_validate_json(usage.model_dump_json()) == usage


def test_all_reported_provider_fields_survive_normalization_round_trip() -> None:
    raw = {
        "input_tokens": 100,
        "output_tokens": 20,
        "reasoning_tokens": 7,
        "cached_input_tokens": 40,
        "provider_total_tokens": 120,
        "provider_request_id": "response-123",
        "measurement": "provider",
        "measurement_source": "provider.response.usage",
    }
    usage = TokenUsage.model_validate(raw)
    assert usage.model_dump() == raw
    assert TokenUsage.model_validate_json(usage.model_dump_json()) == usage


def test_explicit_zero_is_known_and_distinct_from_missing_usage() -> None:
    usage = TokenUsage(
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cached_input_tokens=0,
        provider_total_tokens=0,
        measurement="provider",
    )
    ledger = BudgetLedger()
    ledger.model_responded(ledger.model_called(), usage)
    totals = ledger.totals(0)
    for dimension in TOKEN_DIMENSIONS:
        assert getattr(totals, dimension) == 0
    assert totals.unknown_reasoning_calls == totals.unknown_cached_input_calls == 0
    assert totals.unknown_provider_total_calls == 0


@pytest.mark.parametrize("dimension", TOKEN_DIMENSIONS)
@pytest.mark.parametrize("invalid", [-1, True, 1.5, "3"])
def test_token_dimensions_reject_invalid_unit_values(dimension: str, invalid: Any) -> None:
    with pytest.raises(ValidationError):
        TokenUsage.model_validate({dimension: invalid, "measurement": "provider"})


@pytest.mark.parametrize("dimension", TOKEN_DIMENSIONS)
def test_known_dimension_cannot_be_mislabeled_unavailable(dimension: str) -> None:
    with pytest.raises(ValidationError, match="measurement source"):
        TokenUsage.model_validate({dimension: 0, "measurement": "unavailable"})


@pytest.mark.parametrize("field", ["provider_request_id", "measurement_source"])
@pytest.mark.parametrize("invalid", ["", "   ", 1])
def test_usage_metadata_rejects_blank_or_nonstring_values(field: str, invalid: Any) -> None:
    with pytest.raises(ValidationError):
        TokenUsage.model_validate({field: invalid})


def test_request_identifier_can_be_known_when_usage_is_absent() -> None:
    usage = TokenUsage(provider_request_id="response-unknown-usage")
    assert usage.measurement == "unavailable"
    assert all(getattr(usage, dimension) is None for dimension in TOKEN_DIMENSIONS)


def test_partial_additional_dimensions_do_not_invent_input_or_output_usage() -> None:
    usage = TokenUsage(
        reasoning_tokens=5,
        cached_input_tokens=10,
        provider_total_tokens=30,
        measurement="provider",
    )
    assert usage.input_tokens is usage.output_tokens is None
    ledger = BudgetLedger()
    ledger.model_responded(ledger.model_called(), usage)
    totals = ledger.totals(0)
    assert totals.input_tokens is totals.output_tokens is None
    assert totals.reasoning_tokens == 5
    assert totals.cached_input_tokens == 10
    assert totals.provider_total_tokens == 30
    assert totals.unknown_input_calls == totals.unknown_output_calls == 1


def test_cache_count_cannot_exceed_known_input_count() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        TokenUsage(input_tokens=5, cached_input_tokens=6, measurement="provider")


def test_raw_provider_total_and_reasoning_are_not_recomputed() -> None:
    usage = TokenUsage(
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=9,
        provider_total_tokens=24,
        measurement="provider",
    )
    ledger = BudgetLedger()
    ledger.model_responded(ledger.model_called(), usage)
    totals = ledger.totals(0)
    assert totals.input_tokens == 10 and totals.output_tokens == 5
    assert totals.reasoning_tokens == 9 and totals.provider_total_tokens == 24


def test_mixed_known_unknown_and_failed_calls_preserve_role_subtotals() -> None:
    ledger = BudgetLedger()
    full = ledger.model_called("weak")
    ledger.model_responded(
        full,
        TokenUsage(
            input_tokens=100,
            output_tokens=20,
            reasoning_tokens=7,
            cached_input_tokens=40,
            provider_total_tokens=120,
            measurement="provider",
        ),
    )
    partial = ledger.model_called("weak")
    ledger.model_responded(
        partial,
        TokenUsage(input_tokens=30, output_tokens=10, reasoning_tokens=0, measurement="provider"),
    )
    teacher = ledger.model_called("teacher")
    ledger.model_responded(
        teacher,
        TokenUsage(
            input_tokens=50,
            output_tokens=12,
            reasoning_tokens=4,
            cached_input_tokens=0,
            provider_total_tokens=62,
            measurement="provider",
        ),
    )
    ledger.model_called("teacher")  # Failed dispatch retains unknown usage in every dimension.
    totals = ledger.totals(0)
    assert totals.model_calls == 4
    assert totals.input_tokens is totals.output_tokens is None
    assert (
        totals.reasoning_tokens
        is totals.cached_input_tokens
        is totals.provider_total_tokens
        is None
    )
    assert totals.known_input_tokens == 180 and totals.known_output_tokens == 42
    assert totals.known_reasoning_tokens == 11
    assert totals.known_cached_input_tokens == 40
    assert totals.known_provider_total_tokens == 182
    assert totals.unknown_reasoning_calls == 1
    assert totals.unknown_cached_input_calls == totals.unknown_provider_total_calls == 2
    assert totals.weak.input_tokens == 130 and totals.weak.output_tokens == 30
    assert totals.weak.reasoning_tokens == 7
    assert totals.weak.cached_input_tokens is totals.weak.provider_total_tokens is None
    assert totals.weak.known_cached_input_tokens == 40
    assert totals.weak.known_provider_total_tokens == 120
    assert totals.teacher.known_reasoning_tokens == 4
    assert totals.teacher.unknown_reasoning_calls == totals.teacher.unknown_cached_input_calls == 1
    assert totals.teacher.unknown_provider_total_calls == 1
    for dimension in TOKEN_DIMENSIONS:
        known = f"known_{dimension}"
        assert getattr(totals, known) == getattr(totals.weak, known) + getattr(
            totals.teacher, known
        )


def test_unknown_additional_dimensions_do_not_null_known_input_output_totals() -> None:
    ledger = BudgetLedger()
    ledger.model_responded(
        ledger.model_called(),
        TokenUsage(input_tokens=8, output_tokens=3, measurement="synthetic"),
    )
    totals = ledger.totals(0)
    assert totals.input_tokens == 8 and totals.output_tokens == 3
    assert totals.synthetic_usage_calls == 1
    assert (
        totals.reasoning_tokens
        is totals.cached_input_tokens
        is totals.provider_total_tokens
        is None
    )
    assert totals.known_reasoning_tokens == totals.known_cached_input_tokens == 0
    assert totals.known_provider_total_tokens == 0
    assert totals.unknown_reasoning_calls == totals.unknown_cached_input_calls == 1
    assert totals.unknown_provider_total_calls == 1


def test_zero_model_calls_have_known_zero_aggregate_usage() -> None:
    totals = BudgetLedger().totals(0)
    for dimension in TOKEN_DIMENSIONS:
        assert getattr(totals, dimension) == 0
        assert getattr(totals.teacher, dimension) == 0
        assert getattr(totals.weak, dimension) == 0
    assert totals.unknown_reasoning_calls == totals.unknown_cached_input_calls == 0
    assert totals.unknown_provider_total_calls == 0


def test_legacy_budget_loads_without_claiming_missing_dimensions_are_known_zero() -> None:
    legacy = {
        "model_calls": 2,
        "input_tokens": 10,
        "output_tokens": 4,
        "known_input_tokens": 10,
        "known_output_tokens": 4,
        "unknown_input_calls": 0,
        "unknown_output_calls": 0,
        "synthetic_usage_calls": 2,
        "wall_clock_ms": 1.5,
    }
    totals = RoleBudgetTotals.model_validate(legacy)
    assert (
        totals.reasoning_tokens
        is totals.cached_input_tokens
        is totals.provider_total_tokens
        is None
    )
    assert totals.known_reasoning_tokens == totals.known_cached_input_tokens == 0
    assert totals.known_provider_total_tokens == 0
    assert totals.unknown_reasoning_calls == totals.unknown_cached_input_calls == 2
    assert totals.unknown_provider_total_calls == 2
    assert RoleBudgetTotals.model_validate_json(totals.model_dump_json()) == totals


def test_new_budget_defaults_preserve_explicit_reported_dimensions() -> None:
    totals = RoleBudgetTotals(
        model_calls=1,
        input_tokens=10,
        output_tokens=5,
        known_input_tokens=10,
        known_output_tokens=5,
        unknown_input_calls=0,
        unknown_output_calls=0,
        synthetic_usage_calls=0,
        wall_clock_ms=0,
        reasoning_tokens=2,
        cached_input_tokens=0,
        provider_total_tokens=15,
    )
    assert totals.known_reasoning_tokens == 2
    assert totals.known_cached_input_tokens == 0
    assert totals.known_provider_total_tokens == 15
    assert totals.unknown_reasoning_calls == totals.unknown_cached_input_calls == 0
    assert totals.unknown_provider_total_calls == 0


def test_repeated_response_cannot_overwrite_recorded_provider_usage() -> None:
    ledger = BudgetLedger()
    call = ledger.model_called()
    original = TokenUsage(input_tokens=10, output_tokens=5, measurement="provider")
    ledger.model_responded(call, original)
    with pytest.raises(ValueError, match="already recorded"):
        ledger.model_responded(call, TokenUsage(reasoning_tokens=0, measurement="provider"))
    assert ledger.totals(0).input_tokens == 10


def test_fake_provider_keeps_new_dimensions_unknown() -> None:
    response = asyncio.run(
        FakeModelProvider("first_available").generate(
            ModelRequest(
                model="fixture",
                system_prompt="Choose an action",
                observation=Observation(agent_id="agent_0", text="{}"),
                available_actions=(ActionSpec(name="wait", description="Wait"),),
                temperature=0,
                max_output_tokens=64,
            )
        )
    )
    usage = response.usage
    assert usage.measurement == "synthetic"
    assert usage.input_tokens is not None and usage.output_tokens is not None
    assert (
        usage.reasoning_tokens is usage.cached_input_tokens is usage.provider_total_tokens is None
    )


def test_learning_phase_merge_preserves_partial_new_dimensions() -> None:
    known, unknown = BudgetLedger(), BudgetLedger()
    call = known.model_called("teacher")
    known.model_responded(
        call,
        TokenUsage(
            input_tokens=9,
            output_tokens=5,
            reasoning_tokens=2,
            cached_input_tokens=0,
            provider_total_tokens=14,
            measurement="provider",
        ),
    )
    unknown.model_called("teacher")
    merged = merge_budgets((known.totals(1), unknown.totals(2)))
    for totals in (merged, merged.teacher):
        assert totals.model_calls == 2
        assert totals.reasoning_tokens is totals.cached_input_tokens is None
        assert totals.provider_total_tokens is None
        assert totals.known_reasoning_tokens == 2
        assert totals.known_cached_input_tokens == 0
        assert totals.known_provider_total_tokens == 14
        assert totals.unknown_reasoning_calls == totals.unknown_cached_input_calls == 1
        assert totals.unknown_provider_total_calls == 1
    assert merged.weak.model_calls == 0 and merged.weak.reasoning_tokens == 0


@pytest.mark.parametrize(
    "field",
    [
        "reasoning_tokens",
        "cached_input_tokens",
        "provider_total_tokens",
        "known_reasoning_tokens",
        "known_cached_input_tokens",
        "known_provider_total_tokens",
        "unknown_reasoning_calls",
        "unknown_cached_input_calls",
        "unknown_provider_total_calls",
    ],
)
def test_validation_cannot_hide_teacher_usage_in_new_dimensions(field: str) -> None:
    budget = BudgetLedger().totals(0).model_dump(mode="json")
    budget["teacher"][field] = 1
    with pytest.raises(ValidationError, match="must not use the teacher"):
        ValidationTrial.model_validate_json(
            json.dumps(
                {
                    "seed": 19,
                    "config_sha256": "0" * 64,
                    "trace_sha256": "1" * 64,
                    "provider": "fake",
                    "model": "fixture",
                    "status": "succeeded",
                    "reason": "verified",
                    "skill_completed": True,
                    "budget": budget,
                }
            )
        )
