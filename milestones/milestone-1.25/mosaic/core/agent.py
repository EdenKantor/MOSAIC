from mosaic.core.config import ExperimentConfig, ModelConfig
from mosaic.core.models import (
    Action,
    ActionSpec,
    AttemptFeedback,
    ModelRequest,
    ModelRole,
    Observation,
)

SYSTEM_PROMPT = (
    'Choose exactly one advertised action. Return only JSON: {"name":"action name"}. '
    "The environment verifies task completion."
)


class Agent:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def request(
        self,
        observation: Observation,
        actions: tuple[ActionSpec, ...],
        *,
        model: ModelConfig | None = None,
        model_role: ModelRole = "weak",
        history: tuple[AttemptFeedback, ...] = (),
        step: int = 0,
        system_prompt: str | None = None,
    ) -> ModelRequest:
        identity = model or self.config.model
        return ModelRequest(
            model=identity.model,
            system_prompt=system_prompt or SYSTEM_PROMPT,
            observation=observation,
            available_actions=actions,
            temperature=identity.temperature,
            max_output_tokens=identity.max_output_tokens,
            model_role=model_role,
            goal=self.config.task,
            history=history,
            step=step,
            thinking=identity.thinking,
            reasoning_effort=identity.reasoning_effort,
            context_window=identity.context_window,
            thinking_budget=identity.thinking_budget,
            thinking_level=identity.thinking_level,
        )

    def parse(self, text: str, actions: tuple[ActionSpec, ...]) -> Action:
        action = Action.model_validate_json(text)
        if action.name not in {spec.name for spec in actions}:
            raise ValueError(f"Action is not advertised: {action.name}")
        return action
