"""Single-request HTTP transport and shared provider serialization helpers."""

import asyncio
import http.client
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from pydantic import ValidationError

from mosaic.core.models import ModelRequest, TokenUsage


class ProviderError(RuntimeError):
    """Static provider failure messages never contain response bodies or credentials."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: dict[str, Any]
    request_id: str | None = None


class HttpTransport(Protocol):
    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> HttpResponse: ...


def redact(text: str, *captured_secrets: str) -> str:
    """Remove captured and current Groq credentials before text reaches artifacts."""
    secrets = {*captured_secrets, os.environ.get("GROQ_API_KEY", "")}
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return text


def visible_messages(request: ModelRequest) -> list[dict[str, str]]:
    history = []
    for feedback in request.history:
        visible_feedback = feedback.model_dump(mode="json", exclude={"model_role"})
        visible_feedback["observation"] = {"text": feedback.observation.text}
        history.append(visible_feedback)
    visible = {
        "goal": request.goal,
        "step": request.step,
        "observation": {"text": request.observation.text},
        "available_actions": [
            action.model_dump(mode="json") for action in request.available_actions
        ],
        "history": history,
    }
    return [
        {"role": "system", "content": request.system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                visible, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        },
    ]


def optional_count(data: dict[str, Any], field: str) -> int | None:
    value = data.get(field)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ProviderError("Provider returned invalid token usage")
    return value


def optional_identifier(value: Any, *secrets: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProviderError("Provider returned invalid response metadata")
    return redact(value, *secrets)


def usage_record(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    reasoning_tokens: int | None,
    cached_input_tokens: int | None,
    provider_total_tokens: int | None,
    request_id: str | None,
    source: str,
) -> TokenUsage:
    known = any(
        value is not None
        for value in (
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cached_input_tokens,
            provider_total_tokens,
        )
    )
    result: TokenUsage | None = None
    try:
        result = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_input_tokens=cached_input_tokens,
            provider_total_tokens=provider_total_tokens,
            provider_request_id=request_id,
            measurement="provider" if known else "unavailable",
            measurement_source=source,
        )
    except ValidationError:
        pass
    if result is None:
        # Raise outside the handler so the validation error and raw input are not chained.
        raise ProviderError("Provider returned inconsistent token usage")
    return result


async def single_request(
    transport: HttpTransport,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> HttpResponse:
    response: HttpResponse | None = None
    try:
        response = await transport.post_json(url, payload, headers, timeout)
    except Exception:
        pass
    if response is None:
        raise ProviderError("Provider request failed")
    if type(response.status) is not int or not 100 <= response.status <= 599:
        raise ProviderError("Provider returned invalid HTTP status")
    if not 200 <= response.status < 300:
        raise ProviderError(f"Provider HTTP request failed with status {response.status}")
    if not isinstance(response.body, dict):
        raise ProviderError("Provider returned invalid response JSON")
    return response


class StdlibHttpTransport:
    """A fresh connection, one POST and no retry or redirect handling per invocation."""

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> HttpResponse:
        result: HttpResponse | None = None
        try:
            result = await asyncio.to_thread(self._post_json, url, payload, headers, timeout)
        except Exception:
            pass
        if result is None:
            raise ProviderError("HTTP transport failed")
        return result

    @staticmethod
    def _post_json(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> HttpResponse:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ProviderError("HTTP transport URL is invalid")
        connection_type = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
        try:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            connection.request(
                "POST",
                parsed.path or "/",
                body=body,
                headers={"Content-Type": "application/json", **headers},
            )
            response = connection.getresponse()
            request_id = response.getheader("x-request-id") or response.getheader(
                "x-groq-request-id"
            )
            # Error and redirect bodies never enter normalization or an exception message.
            if not 200 <= response.status < 300:
                return HttpResponse(status=response.status, body={}, request_id=request_id)
            decoded = json.loads(response.read().decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ProviderError("HTTP response JSON must be an object")
            return HttpResponse(status=response.status, body=decoded, request_id=request_id)
        finally:
            connection.close()
