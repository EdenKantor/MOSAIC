"""Local Ollama chat adapter; only visible action context enters the request."""

import ipaddress
import os
from typing import Any
from urllib.parse import urlsplit

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


def _local_endpoint() -> str:
    configured = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    valid = False
    try:
        parsed = urlsplit(configured)
        host = parsed.hostname
        loopback = host == "localhost"
        if host is not None and not loopback:
            loopback = ipaddress.ip_address(host).is_loopback
        valid = (
            parsed.scheme in {"http", "https"}
            and loopback
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
            and not any(character.isspace() for character in configured)
            and (parsed.port is None or 1 <= parsed.port <= 65535)
            and "?" not in configured
            and "#" not in configured
        )
    except ValueError:
        pass
    if not valid:
        raise ProviderError("OLLAMA_HOST must be a loopback HTTP URL without credentials")
    return configured.rstrip("/") + "/api/chat"


class OllamaModelProvider:
    def __init__(
        self, timeout_seconds: float = 120, *, transport: HttpTransport | None = None
    ) -> None:
        self.__endpoint = _local_endpoint()
        self.__transport = transport or StdlibHttpTransport()
        self.__timeout_seconds = timeout_seconds
        self.__captured_secrets = environment_secrets()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "num_predict": request.max_output_tokens,
        }
        if request.context_window is not None:
            options["num_ctx"] = request.context_window
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": visible_messages(request),
            "stream": False,
            "format": "json",
            "options": options,
        }
        if request.thinking is not None:
            payload["think"] = request.thinking
        response = await single_request(
            self.__transport, self.__endpoint, payload, {}, self.__timeout_seconds
        )
        body = response.body
        message = body.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderError("Ollama returned invalid assistant content")
        model = body.get("model", request.model)
        finish = body.get("done_reason", "unknown")
        if not isinstance(model, str) or not isinstance(finish, str):
            raise ProviderError("Ollama returned invalid response metadata")
        identifier = optional_identifier(
            body.get("request_id", response.request_id), *self.__captured_secrets
        )
        usage = usage_record(
            input_tokens=optional_count(body, "prompt_eval_count"),
            output_tokens=optional_count(body, "eval_count"),
            reasoning_tokens=optional_count(body, "reasoning_tokens"),
            cached_input_tokens=optional_count(body, "prompt_eval_cached_count"),
            provider_total_tokens=optional_count(body, "total_tokens"),
            request_id=identifier,
            source="ollama.api.chat",
        )
        return ModelResponse(
            provider="ollama",
            model=redact(model, *self.__captured_secrets),
            text=redact(message["content"], *self.__captured_secrets),
            usage=usage,
            finish_reason=redact(finish, *self.__captured_secrets),
            reported_model=optional_identifier(body.get("model"), *self.__captured_secrets),
            rate_limits=response.rate_limits,
        )
