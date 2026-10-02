from dataclasses import dataclass

from pydantic import Field

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


def _totals(calls: list[_CallUsage], wall_clock_ms: float) -> RoleBudgetTotals:
    unknown_in = sum(c.usage.input_tokens is None for c in calls)
    unknown_out = sum(c.usage.output_tokens is None for c in calls)
    known_in = sum(c.usage.input_tokens or 0 for c in calls)
    known_out = sum(c.usage.output_tokens or 0 for c in calls)
    return RoleBudgetTotals(
        model_calls=len(calls),
        input_tokens=None if unknown_in else known_in,
        output_tokens=None if unknown_out else known_out,
        known_input_tokens=known_in,
        known_output_tokens=known_out,
        unknown_input_calls=unknown_in,
        unknown_output_calls=unknown_out,
        synthetic_usage_calls=sum(c.usage.measurement == "synthetic" for c in calls),
        wall_clock_ms=wall_clock_ms,
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
