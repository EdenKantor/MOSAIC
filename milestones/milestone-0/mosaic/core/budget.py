from pydantic import Field

from mosaic.core.models import FrozenModel, TokenUsage


class BudgetTotals(FrozenModel):
    model_calls: int = Field(ge=0)
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    known_input_tokens: int = Field(ge=0)
    known_output_tokens: int = Field(ge=0)
    unknown_input_calls: int = Field(ge=0)
    unknown_output_calls: int = Field(ge=0)
    synthetic_usage_calls: int = Field(ge=0)
    environment_actions: int = Field(ge=0)
    wall_clock_ms: float = Field(ge=0)


class BudgetLedger:
    """Every dispatch reserves a usage entry, even if the provider raises."""

    def __init__(self) -> None:
        self._usage: list[TokenUsage] = []
        self._environment_actions = 0

    def model_called(self) -> int:
        self._usage.append(TokenUsage())
        return len(self._usage) - 1

    def model_responded(self, call_index: int, usage: TokenUsage) -> None:
        self._usage[call_index] = usage

    def action_called(self) -> None:
        self._environment_actions += 1

    def totals(self, wall_clock_ms: float) -> BudgetTotals:
        unknown_in = sum(u.input_tokens is None for u in self._usage)
        unknown_out = sum(u.output_tokens is None for u in self._usage)
        known_in = sum(u.input_tokens or 0 for u in self._usage)
        known_out = sum(u.output_tokens or 0 for u in self._usage)
        return BudgetTotals(
            model_calls=len(self._usage),
            input_tokens=None if unknown_in else known_in,
            output_tokens=None if unknown_out else known_out,
            known_input_tokens=known_in,
            known_output_tokens=known_out,
            unknown_input_calls=unknown_in,
            unknown_output_calls=unknown_out,
            synthetic_usage_calls=sum(u.measurement == "synthetic" for u in self._usage),
            environment_actions=self._environment_actions,
            wall_clock_ms=wall_clock_ms,
        )
