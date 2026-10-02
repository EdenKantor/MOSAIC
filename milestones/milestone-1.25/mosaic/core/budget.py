from dataclasses import dataclass
from typing import Any

from pydantic import Field, model_validator

from mosaic.core.models import FrozenModel, ModelRole, TokenUsage


class RoleBudgetTotals(FrozenModel):
    model_calls: int = Field(ge=0)
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    known_input_tokens: int = Field(ge=0)
    known_output_tokens: int = Field(ge=0)
    unknown_input_calls: int = Field(ge=0)
    unknown_output_calls: int = Field(ge=0)
    synthetic_usage_calls: int = Field(ge=0)
    wall_clock_ms: float = Field(ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    provider_total_tokens: int | None = Field(default=None, ge=0)
    known_reasoning_tokens: int = Field(default=0, ge=0)
    known_cached_input_tokens: int = Field(default=0, ge=0)
    known_provider_total_tokens: int = Field(default=0, ge=0)
    unknown_reasoning_calls: int = Field(default=0, ge=0)
    unknown_cached_input_calls: int = Field(default=0, ge=0)
    unknown_provider_total_calls: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def preserve_legacy_unknown_usage(cls, value: Any) -> Any:
        """A legacy summary omitting a dimension supplies no evidence of zero usage."""
        if not isinstance(value, dict):
            return value
        result = dict(value)
        for dimension, unknown_field in (
            ("reasoning_tokens", "unknown_reasoning_calls"),
            ("cached_input_tokens", "unknown_cached_input_calls"),
            ("provider_total_tokens", "unknown_provider_total_calls"),
        ):
            reported = result.get(dimension)
            if unknown_field not in result:
                result[unknown_field] = result.get("model_calls", 0) if reported is None else 0
            known_field = f"known_{dimension}"
            if known_field not in result and reported is not None:
                result[known_field] = reported
        return result


class BudgetTotals(RoleBudgetTotals):
    environment_actions: int = Field(ge=0)
    weak: RoleBudgetTotals
    teacher: RoleBudgetTotals


@dataclass
class _CallUsage:
    role: ModelRole
    usage: TokenUsage
    wall_clock_ms: float = 0
    responded: bool = False


def _dimension_totals(calls: list[_CallUsage], dimension: str) -> tuple[int | None, int, int]:
    values: list[int | None] = [getattr(call.usage, dimension) for call in calls]
    unknown = sum(value is None for value in values)
    known = sum(value for value in values if value is not None)
    return None if unknown else known, known, unknown


def _totals(calls: list[_CallUsage], wall_clock_ms: float) -> RoleBudgetTotals:
    input_tokens, known_in, unknown_in = _dimension_totals(calls, "input_tokens")
    output_tokens, known_out, unknown_out = _dimension_totals(calls, "output_tokens")
    reasoning, known_reasoning, unknown_reasoning = _dimension_totals(calls, "reasoning_tokens")
    cached, known_cached, unknown_cached = _dimension_totals(calls, "cached_input_tokens")
    provider_total, known_total, unknown_total = _dimension_totals(calls, "provider_total_tokens")
    return RoleBudgetTotals(
        model_calls=len(calls),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        known_input_tokens=known_in,
        known_output_tokens=known_out,
        unknown_input_calls=unknown_in,
        unknown_output_calls=unknown_out,
        synthetic_usage_calls=sum(c.usage.measurement == "synthetic" for c in calls),
        wall_clock_ms=wall_clock_ms,
        reasoning_tokens=reasoning,
        cached_input_tokens=cached,
        provider_total_tokens=provider_total,
        known_reasoning_tokens=known_reasoning,
        known_cached_input_tokens=known_cached,
        known_provider_total_tokens=known_total,
        unknown_reasoning_calls=unknown_reasoning,
        unknown_cached_input_calls=unknown_cached,
        unknown_provider_total_calls=unknown_total,
    )


class BudgetLedger:
    """Per-role call reservations count failures with unknown usage, never zero tokens."""

    def __init__(self) -> None:
        self._calls: list[_CallUsage] = []
        self._environment_actions = 0

    def model_called(self, role: ModelRole = "weak") -> int:
        self._calls.append(_CallUsage(role=role, usage=TokenUsage()))
        return len(self._calls) - 1

    def _call(self, call_index: int) -> _CallUsage:
        if call_index < 0 or call_index >= len(self._calls):
            raise ValueError("Unknown model call reservation")
        return self._calls[call_index]

    def model_responded(self, call_index: int, usage: TokenUsage) -> None:
        call = self._call(call_index)
        if call.responded:
            raise ValueError("Usage already recorded for this model call")
        call.usage, call.responded = usage, True

    def model_finished(self, call_index: int, latency_ms: float) -> None:
        if latency_ms < 0:
            raise ValueError("Latency cannot be negative")
        self._call(call_index).wall_clock_ms = latency_ms

    def action_called(self) -> None:
        self._environment_actions += 1

    def totals(self, wall_clock_ms: float) -> BudgetTotals:
        weak = [call for call in self._calls if call.role == "weak"]
        teacher = [call for call in self._calls if call.role == "teacher"]
        return BudgetTotals(
            **_totals(self._calls, wall_clock_ms).model_dump(),
            environment_actions=self._environment_actions,
            weak=_totals(weak, sum(call.wall_clock_ms for call in weak)),
            teacher=_totals(teacher, sum(call.wall_clock_ms for call in teacher)),
        )
