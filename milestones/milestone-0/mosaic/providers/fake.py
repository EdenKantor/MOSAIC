import json
from typing import Literal

from mosaic.core.models import Action, ModelRequest, ModelResponse, TokenUsage


class FakeModelProvider:
    """Deterministic fixture policy, not an LLM or evidence of agent learning."""

    def __init__(self, policy: Literal["repair_pump", "first_available"] = "repair_pump") -> None:
        self.policy = policy

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if not request.available_actions:
            raise ValueError("No action available")
        if self.policy == "first_available":
            name = request.available_actions[0].name
        else:
            state = json.loads(request.observation.text)
            if state["has_part"]:
                name = "repair" if state["position"] == 0 else "move_west"
            else:
                name = "collect" if state["position"] == state["part_position"] else "move_east"
        text = Action(name=name).model_dump_json()
        # These whitespace counts are explicitly synthetic fixture units.
        input_text = request.model_dump_json()
        return ModelResponse(
            provider="fake",
            model=request.model,
            text=text,
            usage=TokenUsage(
                input_tokens=len(input_text.split()),
                output_tokens=len(text.split()),
                measurement="synthetic",
            ),
            finish_reason="stop",
        )
