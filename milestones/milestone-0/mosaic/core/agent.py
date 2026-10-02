from mosaic.core.config import ExperimentConfig
from mosaic.core.models import Action, ActionSpec, ModelRequest, Observation

SYSTEM_PROMPT = (
    'Choose exactly one advertised action. Return only JSON: {"name":"action name"}. '
    "The environment verifies task completion."
)


class Agent:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def request(self, observation: Observation, actions: tuple[ActionSpec, ...]) -> ModelRequest:
        return ModelRequest(
            model=self.config.model.model,
            system_prompt=SYSTEM_PROMPT,
            observation=observation,
            available_actions=actions,
            temperature=self.config.model.temperature,
            max_output_tokens=self.config.model.max_output_tokens,
        )

    def parse(self, text: str, actions: tuple[ActionSpec, ...]) -> Action:
        action = Action.model_validate_json(text)
        if action.name not in {spec.name for spec in actions}:
            raise ValueError(f"Action is not advertised: {action.name}")
        return action
