"""Gemini Developer API structured actions, with one physical request and no retry."""

import os
import re
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


class GeminiModelProvider:
    def __init__(
        self, timeout_seconds: float = 120, *, transport: HttpTransport | None = None
    ) -> None:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key.strip() or "\r" in key or "\n" in key:
            raise ProviderError(
                "GEMINI_API_KEY is required in the environment", category="authentication"
            )
        self.__api_key = key
        self.__captured_secrets = environment_secrets()
        self.__timeout_seconds = timeout_seconds
        self.__transport = transport or StdlibHttpTransport()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        model_id = request.model.removeprefix("models/")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_id) is None:
            raise ProviderError("Gemini model identifier is invalid")
        if not request.available_actions:
            raise ProviderError("Gemini generation requires advertised actions")
        messages = visible_messages(request)
        generation: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": request.max_output_tokens,
            "candidateCount": 1,
            "responseMimeType": "application/json",
            "thinkingConfig": {"includeThoughts": False},
        }
        # Common JSON mode leaves action legality to the same local validator on all providers.
        if request.thinking_budget is not None:
            generation["thinkingConfig"]["thinkingBudget"] = request.thinking_budget
        if request.thinking_level is not None:
            generation["thinkingConfig"]["thinkingLevel"] = request.thinking_level.upper()
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
            "contents": [{"role": "user", "parts": [{"text": messages[1]["content"]}]}],
            "generationConfig": generation,
        }
        response = await single_request(
            self.__transport,
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
            payload,
            {"x-goog-api-key": self.__api_key},
            self.__timeout_seconds,
        )
        body = response.body
        candidates = body.get("candidates")
        if (
            not isinstance(candidates, list)
            or len(candidates) != 1
            or not isinstance(candidates[0], dict)
        ):
            raise ProviderError("Gemini returned invalid response candidates")
        candidate = candidates[0]
        content = candidate.get("content")
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
            raise ProviderError("Gemini returned invalid assistant content")
        texts: list[str] = []
        for part in content["parts"]:
            if not isinstance(part, dict):
                raise ProviderError("Gemini returned invalid content parts")
            if part.get("thought") is True:
                continue
            text = part.get("text")
            if text is not None:
                if not isinstance(text, str):
                    raise ProviderError("Gemini returned invalid assistant text")
                texts.append(text)
        finish = candidate.get("finishReason", "unknown")
        if not isinstance(finish, str):
            raise ProviderError("Gemini returned invalid finish metadata")
        reported = body.get("usageMetadata")
        if reported is not None and not isinstance(reported, dict):
            raise ProviderError("Gemini returned invalid token usage")
        reported = reported or {}
        identifier = optional_identifier(
            body.get("responseId", response.request_id), *self.__captured_secrets
        )
        reported_model = optional_identifier(body.get("modelVersion"), *self.__captured_secrets)
        usage = usage_record(
            input_tokens=optional_count(reported, "promptTokenCount"),
            output_tokens=optional_count(reported, "candidatesTokenCount"),
            reasoning_tokens=optional_count(reported, "thoughtsTokenCount"),
            cached_input_tokens=optional_count(reported, "cachedContentTokenCount"),
            provider_total_tokens=optional_count(reported, "totalTokenCount"),
            request_id=identifier,
            source="gemini.generateContent.usageMetadata",
        )
        return ModelResponse(
            provider="gemini",
            model=redact(request.model, *self.__captured_secrets),
            reported_model=reported_model,
            text=redact("".join(texts), *self.__captured_secrets),
            usage=usage,
            finish_reason=redact(finish, *self.__captured_secrets),
            rate_limits=response.rate_limits,
        )
