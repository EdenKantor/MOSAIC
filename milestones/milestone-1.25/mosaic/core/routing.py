from dataclasses import dataclass
from typing import Literal, Protocol

from mosaic.core.config import RoutingConfig
from mosaic.core.models import FrozenModel


@dataclass(frozen=True)
class RoutingState:
    phase: Literal["attempt", "recovery"]
    phase_steps: int
    recoveries_started: int
    consecutive_failures: int
    remaining_steps: int
    succeeded: bool
    environment_done: bool


class RoutingDecision(FrozenModel):
    outcome: Literal["continue_weak", "start_recovery", "escalate", "terminate"]
    reason: str


class EscalationPolicy(Protocol):
    def decide(self, state: RoutingState) -> RoutingDecision: ...


class BoundedFailurePolicy:
    """Objective failures or a finite success deadline, never model confidence."""

    def __init__(self, config: RoutingConfig) -> None:
        self.config = config

    def decide(self, state: RoutingState) -> RoutingDecision:
        if state.succeeded:
            return RoutingDecision(outcome="terminate", reason="environment_verified")
        if state.environment_done:
            return RoutingDecision(outcome="terminate", reason="environment_done")
        if state.remaining_steps <= 0:
            return RoutingDecision(outcome="terminate", reason="max_steps")
        limit = (
            self.config.attempt_steps if state.phase == "attempt" else self.config.recovery_steps
        )
        if state.consecutive_failures >= self.config.consecutive_failure_limit:
            reason = "consecutive_execution_failures"
        elif state.phase_steps >= limit:
            reason = "weak_window_exhausted_without_goal"
        else:
            return RoutingDecision(outcome="continue_weak", reason="within_weak_window")
        if state.recoveries_started < self.config.recovery_attempts:
            return RoutingDecision(outcome="start_recovery", reason=reason)
        return RoutingDecision(outcome="escalate", reason=reason)
