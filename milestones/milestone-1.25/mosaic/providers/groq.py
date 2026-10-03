"""Groq chat completions adapter with environment-only credentials and no retries."""

import os
from typing import Any

from mosaic.core.models import ModelRequest, ModelResponse
from mosaic.providers.http import (
    HttpTransport,
    ProviderError,
    StdlibHttpTransport,
    environment_secrets,
    optional_count,
    optional_identifier,
    redact,
    single_request,
    usage_record,
    visible_messages,
)


class GroqModelProvider:
    def __init__(
        self, timeout_seconds: float = 120, *, transport: HttpTransport | None = None
    ) -> None:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key.strip() or "\r" in key or "\n" in key:
            raise ProviderError(
                "GROQ_API_KEY is required in the environment", category="authentication"
            )
        self.__api_key = key
        self.__captured_secrets = environment_secrets()
        self.__timeout_seconds = timeout_seconds
        self.__transport = transport or StdlibHttpTransport()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": visible_messages(request),
            "temperature": request.temperature,
            "max_completion_tokens": request.max_output_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
            "n": 1,
        }
        if request.reasoning_effort is not None:
            payload["reasoning_effort"] = request.reasoning_effort
        response = await single_request(
            self.__transport,
            "https://api.groq.com/openai/v1/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.__api_key}"},
            self.__timeout_seconds,
        )
        body = response.body
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError("Groq returned invalid completion choices")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderError("Groq returned invalid assistant content")
        model = body.get("model", request.model)
        finish = choice.get("finish_reason", "unknown")
        if not isinstance(model, str) or not isinstance(finish, str):
            raise ProviderError("Groq returned invalid response metadata")
        raw_usage = body.get("usage")
        if raw_usage is not None and not isinstance(raw_usage, dict):
            raise ProviderError("Groq returned invalid token usage")
        reported: dict[str, Any] = raw_usage or {}
        output_details = reported.get("completion_tokens_details")
        input_details = reported.get("prompt_tokens_details")
        if (output_details is not None and not isinstance(output_details, dict)) or (
            input_details is not None and not isinstance(input_details, dict)
        ):
            raise ProviderError("Groq returned invalid token usage details")
        metadata = body.get("x_groq")
        if metadata is not None and not isinstance(metadata, dict):
            raise ProviderError("Groq returned invalid request metadata")
        raw_identifier = response.request_id
        if body.get("id") is not None:
            raw_identifier = body["id"]
        if metadata is not None and metadata.get("id") is not None:
            raw_identifier = metadata["id"]
        identifier = optional_identifier(raw_identifier, *self.__captured_secrets)
        usage = usage_record(
            input_tokens=optional_count(reported, "prompt_tokens"),
            output_tokens=optional_count(reported, "completion_tokens"),
            reasoning_tokens=optional_count(output_details or {}, "reasoning_tokens"),
            cached_input_tokens=optional_count(input_details or {}, "cached_tokens"),
            provider_total_tokens=optional_count(reported, "total_tokens"),
            request_id=identifier,
            source="groq.chat.completions.usage",
        )
        return ModelResponse(
            provider="groq",
            model=redact(model, *self.__captured_secrets),
            text=redact(message["content"], *self.__captured_secrets),
            usage=usage,
            finish_reason=redact(finish, *self.__captured_secrets),
            reported_model=optional_identifier(body.get("model"), *self.__captured_secrets),
            rate_limits=response.rate_limits,
        )
