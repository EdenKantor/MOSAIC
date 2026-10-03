"""Single-request HTTP transport and shared provider serialization helpers."""

import asyncio
import http.client
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from pydantic import ValidationError

from mosaic.core.models import ModelRequest, ProviderRateLimits, TokenUsage

FailureCategory = Literal[
    "rate_limit",
    "quota",
    "billing",
    "payment",
    "insufficient_credit",
    "free_tier_unavailable",
    "timeout",
    "server_error",
    "authentication",
    "access",
    "not_found",
    "redirect",
    "transport",
    "invalid_response",
]

_STOP_CATEGORIES: frozenset[FailureCategory] = frozenset(
    {
        "rate_limit",
        "quota",
        "billing",
        "payment",
        "insufficient_credit",
        "free_tier_unavailable",
        "timeout",
        "server_error",
        "authentication",
        "access",
        "redirect",
        "transport",
    }
)
_FAILURE_CATEGORIES: frozenset[FailureCategory] = _STOP_CATEGORIES | {
    "not_found",
    "invalid_response",
}


class ProviderError(RuntimeError):
    """Static provider failure messages never contain response bodies or credentials."""

    def __init__(
        self,
        message: str,
        *,
        category: FailureCategory = "invalid_response",
        http_status: int | None = None,
        stop_batch: bool | None = None,
    ) -> None:
        super().__init__(redact(message))
        self.category: FailureCategory = (
            category if category in _FAILURE_CATEGORIES else "invalid_response"
        )
        self.http_status = (
            http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        )
        self.stop_batch = (
            self.category in _STOP_CATEGORIES if type(stop_batch) is not bool else stop_batch
        )


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: dict[str, Any]
    request_id: str | None = None
    failure_category: FailureCategory | None = None
    rate_limits: ProviderRateLimits | None = None


class HttpTransport(Protocol):
    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> HttpResponse: ...


def environment_secrets() -> tuple[str, ...]:
    """Capture credentials only in process memory for response redaction."""
    return tuple(os.environ.get(name, "") for name in ("GROQ_API_KEY", "GEMINI_API_KEY"))


def redact(text: str, *captured_secrets: str) -> str:
    """Remove captured and current provider credentials before artifact serialization."""
    secrets = {*captured_secrets, *environment_secrets()}
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return text


def _header_count(value: str | None, *secrets: str) -> int | None:
    if value is None:
        return None
    normalized = redact(value.strip(), *secrets)
    if re.fullmatch(r"[0-9]{1,20}", normalized) is None:
        return None
    return int(normalized)


def _header_duration(value: str | None, *secrets: str) -> float | None:
    if value is None:
        return None
    normalized = redact(value.strip(), *secrets)
    if len(normalized) > 128:
        return None
    pieces = list(re.finditer(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h|d)", normalized))
    if not pieces or "".join(piece.group() for piece in pieces) != normalized:
        return None
    units = {"d": (4, 86400), "h": (3, 3600), "m": (2, 60), "s": (1, 1), "ms": (0, 0.001)}
    previous = 5
    seconds = 0.0
    for piece in pieces:
        order, multiplier = units[piece.group(2)]
        if order >= previous:
            return None
        previous = order
        seconds += float(piece.group(1)) * multiplier
    return seconds if math.isfinite(seconds) else None


def groq_rate_limits(headers: Mapping[str, str | None], *secrets: str) -> ProviderRateLimits | None:
    """Keep only numeric Groq quotas: request headers are RPD, token headers TPM."""
    values: dict[str, int | float | None] = {
        "request_limit": _header_count(headers.get("x-ratelimit-limit-requests"), *secrets),
        "token_limit": _header_count(headers.get("x-ratelimit-limit-tokens"), *secrets),
        "remaining_requests": _header_count(
            headers.get("x-ratelimit-remaining-requests"), *secrets
        ),
        "remaining_tokens": _header_count(headers.get("x-ratelimit-remaining-tokens"), *secrets),
        "request_reset_seconds": _header_duration(
            headers.get("x-ratelimit-reset-requests"), *secrets
        ),
        "token_reset_seconds": _header_duration(headers.get("x-ratelimit-reset-tokens"), *secrets),
    }
    if all(value is None for value in values.values()):
        return None
    return ProviderRateLimits.model_validate(values)


def failure_category(status: int, body: dict[str, Any]) -> FailureCategory:
    """Classify private provider error content without returning any of that content."""
    error = body.get("error")
    codes: set[str] = set()
    if isinstance(error, dict):
        codes = {
            re.sub(r"[\s-]+", "_", value.strip().lower())
            for field in ("status", "code", "type", "reason")
            if isinstance(value := error.get(field), str)
        }
        message = error.get("message")
    else:
        message = error
    # Help/upgrade links are advice, not evidence that billing blocked the request.
    hint = re.sub(r"https?://\S+", "", message.lower()) if isinstance(message, str) else ""
    hint = re.sub(r"[_-]+", " ", hint)
    if codes & {"insufficient_credit", "insufficient_credits", "insufficient_balance"} or re.search(
        r"\binsufficient (?:available )?(?:credits?|balance|funds?)\b", hint
    ):
        return "insufficient_credit"
    if codes & {"billing_required", "billing_disabled", "billing_not_enabled"} or re.search(
        r"\b(?:billing(?: account)?(?: is)? (?:required|disabled|not enabled|unavailable)|"
        r"billing must be enabled|(?:requires?|needs?) (?:an? )?(?:active )?billing(?: account)?|"
        r"no billing account|paid tier(?: is)? required|requires? (?:a )?paid tier)\b",
        hint,
    ):
        return "billing"
    if codes & {"payment_required", "payment_failed"} or re.search(
        r"\bpayment(?: is)? (?:required|failed|declined)\b", hint
    ):
        return "payment"
    if codes & {
        "free_tier_unavailable",
        "free_tier_not_supported",
        "free_tier_disabled",
    } or re.search(
        r"\bfree tier(?: is)? (?:unavailable|not available|not supported|disabled|not eligible)\b",
        hint,
    ):
        return "free_tier_unavailable"
    if "resource_exhausted" in codes:
        return "quota"
    if codes & {"rate_limit_exceeded", "rate_limit_error", "too_many_requests"} or status == 429:
        return "rate_limit"
    if codes & {"quota_exceeded", "quota_exhausted", "insufficient_quota"} or "quota" in hint:
        return "quota"
    if "rate limit" in hint:
        return "rate_limit"
    if status == 402:
        return "payment"
    if status == 401:
        return "authentication"
    if "api key" in hint and any(word in hint for word in ("invalid", "not valid", "expired")):
        return "authentication"
    if status == 403:
        return "access"
    if status == 404:
        return "not_found"
    if 300 <= status < 400:
        return "redirect"
    if status in {408, 504}:
        return "timeout"
    if 500 <= status < 600:
        return "server_error"
    return "invalid_response"


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
    category: FailureCategory = "transport"
    failed_status: int | None = None
    should_stop: bool | None = None
    try:
        response = await transport.post_json(url, payload, headers, timeout)
    except ProviderError as error:
        category, failed_status, should_stop = error.category, error.http_status, error.stop_batch
    except TimeoutError:
        category = "timeout"
    except Exception:
        pass
    if response is None:
        raise ProviderError(
            "Provider request failed",
            category=category,
            http_status=failed_status,
            stop_batch=should_stop,
        )
    if type(response.status) is not int or not 100 <= response.status <= 599:
        raise ProviderError("Provider returned invalid HTTP status")
    if not 200 <= response.status < 300:
        classification = response.failure_category or failure_category(
            response.status, response.body if isinstance(response.body, dict) else {}
        )
        raise ProviderError(
            f"Provider HTTP request failed with status {response.status}",
            category=classification,
            http_status=response.status,
        )
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
        category: FailureCategory = "transport"
        try:
            result = await asyncio.to_thread(self._post_json, url, payload, headers, timeout)
        except TimeoutError:
            category = "timeout"
        except Exception:
            pass
        if result is None:
            raise ProviderError("HTTP transport failed", category=category)
        return result

    @staticmethod
    def _post_json(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> HttpResponse:
        captured_secrets = environment_secrets() + (
            headers.get("x-goog-api-key", ""),
            headers.get("Authorization", "").removeprefix("Bearer "),
        )
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
            request_id = (
                response.getheader("x-request-id")
                or response.getheader("x-groq-request-id")
                or response.getheader("x-goog-request-id")
            )
            if request_id is not None:
                request_id = redact(request_id, *captured_secrets)
            rate_limits = None
            if parsed.hostname == "api.groq.com":
                rate_limits = groq_rate_limits(
                    {
                        name: response.getheader(name)
                        for name in (
                            "x-ratelimit-limit-requests",
                            "x-ratelimit-limit-tokens",
                            "x-ratelimit-remaining-requests",
                            "x-ratelimit-remaining-tokens",
                            "x-ratelimit-reset-requests",
                            "x-ratelimit-reset-tokens",
                        )
                    },
                    *captured_secrets,
                )
            if not 200 <= response.status < 300:
                error_body: dict[str, Any] = {}
                if not 300 <= response.status < 400:
                    try:
                        decoded_error = json.loads(response.read(65536).decode("utf-8"))
                        if isinstance(decoded_error, dict):
                            error_body = decoded_error
                    except Exception:
                        pass
                # Only this safe category survives; raw error JSON never reaches callers.
                classification = failure_category(response.status, error_body)
                return HttpResponse(
                    status=response.status,
                    body={},
                    request_id=request_id,
                    failure_category=classification,
                    rate_limits=rate_limits,
                )
            decoded = json.loads(response.read().decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ProviderError("HTTP response JSON must be an object")
            return HttpResponse(
                status=response.status,
                body=decoded,
                request_id=request_id,
                rate_limits=rate_limits,
            )
        finally:
            connection.close()
