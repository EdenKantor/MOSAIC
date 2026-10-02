import asyncio
import json
from typing import Any

import pytest

from mosaic.core.config import ModelConfig, TeacherAccess
from mosaic.core.models import (
    Action,
    ActionSpec,
    AttemptFeedback,
    ModelRequest,
    ModelResponse,
    Observation,
)
from mosaic.core.teacher import TeacherCapability, TeacherUnavailable
from mosaic.providers.factory import provider_factory
from mosaic.providers.fake import FakeModelProvider
from mosaic.providers.groq import GroqModelProvider
from mosaic.providers.http import HttpResponse, ProviderError, StdlibHttpTransport, redact
from mosaic.providers.ollama import OllamaModelProvider

FAKE_SECRET = "TEST_ONLY_GROQ_CREDENTIAL_123"


class MockTransport:
    def __init__(self, response: HttpResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any], dict[str, str], float]] = []

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> HttpResponse:
        self.calls.append((url, payload, headers, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def request(**updates: Any) -> ModelRequest:
    visible = Observation(agent_id="agent_0", text='{"position":1}')
    return ModelRequest.model_validate(
        {
            "model": "configured-model",
            "system_prompt": 'Return JSON with one advertised action as {"name":"action"}.',
            "observation": visible,
            "available_actions": (ActionSpec(name="wait", description="Wait one step"),),
            "temperature": 0,
            "max_output_tokens": 73,
            "model_role": "teacher",
            "goal": "visible-goal",
            "step": 2,
            "history": (
                AttemptFeedback(
                    step=1,
                    model_role="weak",
                    proposal='{"name":"wait"}',
                    action=Action(name="wait"),
                    accepted=True,
                    feedback="Accepted visible wait",
                    observation=visible,
                ),
            ),
            **updates,
        }
    )


def ollama_body(**updates: Any) -> dict[str, Any]:
    return {
        "model": "configured-model",
        "message": {"content": '{"name":"wait"}', "thinking": "PRIVATE_REASONING_TEXT"},
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 25,
        "prompt_eval_cached_count": 40,
        **updates,
    }


def groq_body(**updates: Any) -> dict[str, Any]:
    return {
        "model": "configured-model",
        "choices": [
            {
                "message": {"content": '{"name":"wait"}', "reasoning": "PRIVATE_REASONING_TEXT"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "total_tokens": 125,
            "completion_tokens_details": {"reasoning_tokens": 7},
            "prompt_tokens_details": {"cached_tokens": 40},
        },
        "x_groq": {"id": "groq-request-1"},
        "id": "completion-1",
        **updates,
    }


@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_SECRET)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


def generate(provider: OllamaModelProvider | GroqModelProvider, **updates: Any) -> ModelResponse:
    return asyncio.run(provider.generate(request(**updates)))


def test_ollama_request_has_explicit_knobs_and_only_visible_context() -> None:
    transport = MockTransport(HttpResponse(200, ollama_body()))
    response = generate(
        OllamaModelProvider(11, transport=transport), thinking=False, context_window=4096
    )
    assert len(transport.calls) == 1
    url, payload, headers, timeout = transport.calls[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert payload["model"] == "configured-model"
    assert payload["stream"] is False and payload["format"] == "json"
    assert payload["think"] is False
    assert payload["options"] == {"temperature": 0, "num_predict": 73, "num_ctx": 4096}
    assert headers == {} and timeout == 11
    messages = payload["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    visible = json.loads(messages[1]["content"])
    assert visible["goal"] == "visible-goal" and visible["step"] == 2
    assert visible["observation"] == {"text": '{"position":1}'}
    assert visible["available_actions"] == [{"name": "wait", "description": "Wait one step"}]
    assert visible["history"][0]["feedback"] == "Accepted visible wait"
    assert "snapshot" not in visible and "verification" not in visible
    assert FAKE_SECRET not in json.dumps(payload)
    assert response.text == '{"name":"wait"}'
    assert "PRIVATE_REASONING_TEXT" not in response.model_dump_json()


def test_groq_request_matches_visible_serialization_without_optional_knob_defaults() -> None:
    ollama_transport = MockTransport(HttpResponse(200, ollama_body()))
    groq_transport = MockTransport(HttpResponse(200, groq_body()))
    generate(OllamaModelProvider(13, transport=ollama_transport))
    generate(GroqModelProvider(13, transport=groq_transport), reasoning_effort="low")
    assert len(groq_transport.calls) == 1
    url, payload, headers, timeout = groq_transport.calls[0]
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert payload["messages"] == ollama_transport.calls[0][1]["messages"]
    assert payload["temperature"] == 0 and payload["max_completion_tokens"] == 73
    assert payload["reasoning_effort"] == "low"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["stream"] is False and payload["n"] == 1
    assert headers == {"Authorization": f"Bearer {FAKE_SECRET}"} and timeout == 13
    assert "think" not in ollama_transport.calls[0][1]
    assert "num_ctx" not in ollama_transport.calls[0][1]["options"]


def test_groq_optional_reasoning_effort_is_omitted_when_unspecified() -> None:
    transport = MockTransport(HttpResponse(200, groq_body()))
    generate(GroqModelProvider(transport=transport))
    assert "reasoning_effort" not in transport.calls[0][1]


def test_ollama_usage_does_not_infer_reasoning_or_provider_totals_from_text() -> None:
    response = generate(
        OllamaModelProvider(transport=MockTransport(HttpResponse(200, ollama_body())))
    )
    usage = response.usage
    assert usage.input_tokens == 100 and usage.output_tokens == 25
    assert usage.cached_input_tokens == 40
    assert usage.reasoning_tokens is usage.provider_total_tokens is None
    assert usage.provider_request_id is None
    assert usage.measurement == "provider" and usage.measurement_source == "ollama.api.chat"


def test_groq_usage_retains_all_reported_dimensions_without_double_counting() -> None:
    response = generate(GroqModelProvider(transport=MockTransport(HttpResponse(200, groq_body()))))
    usage = response.usage
    assert usage.input_tokens == 100 and usage.output_tokens == 25
    assert usage.reasoning_tokens == 7 and usage.cached_input_tokens == 40
    assert usage.provider_total_tokens == 125 and usage.provider_request_id == "groq-request-1"
    assert usage.measurement_source == "groq.chat.completions.usage"


@pytest.mark.parametrize("provider_name", ["ollama", "groq"])
def test_missing_usage_preserves_unknown_dimensions(provider_name: str) -> None:
    provider: OllamaModelProvider | GroqModelProvider
    if provider_name == "ollama":
        body = ollama_body(prompt_eval_count=None, eval_count=None, prompt_eval_cached_count=None)
        provider = OllamaModelProvider(transport=MockTransport(HttpResponse(200, body)))
    else:
        body = groq_body(usage=None)
        provider = GroqModelProvider(transport=MockTransport(HttpResponse(200, body)))
    usage = generate(provider).usage
    assert usage.input_tokens is usage.output_tokens is usage.reasoning_tokens is None
    assert usage.cached_input_tokens is usage.provider_total_tokens is None
    assert usage.measurement == "unavailable"


@pytest.mark.parametrize("invalid", [True, -1, 2.5, "25"])
@pytest.mark.parametrize("provider_name", ["ollama", "groq"])
def test_malformed_provider_token_counts_are_rejected(provider_name: str, invalid: Any) -> None:
    provider: OllamaModelProvider | GroqModelProvider
    if provider_name == "ollama":
        body = ollama_body(eval_count=invalid)
        provider = OllamaModelProvider(transport=MockTransport(HttpResponse(200, body)))
    else:
        body = groq_body()
        body["usage"]["completion_tokens"] = invalid
        provider = GroqModelProvider(transport=MockTransport(HttpResponse(200, body)))
    with pytest.raises(ProviderError, match="invalid token usage"):
        generate(provider)


@pytest.mark.parametrize("provider_name", ["ollama", "groq"])
def test_reported_model_identity_is_preserved_for_runner_validation(provider_name: str) -> None:
    provider: OllamaModelProvider | GroqModelProvider
    if provider_name == "ollama":
        provider = OllamaModelProvider(
            transport=MockTransport(HttpResponse(200, ollama_body(model="actual-served-model")))
        )
    else:
        provider = GroqModelProvider(
            transport=MockTransport(HttpResponse(200, groq_body(model="actual-served-model")))
        )
    assert generate(provider).model == "actual-served-model"


@pytest.mark.parametrize("status", [301, 307, 401, 403, 429, 500])
@pytest.mark.parametrize("provider_name", ["ollama", "groq"])
def test_http_failures_are_sanitized_and_never_retried(status: int, provider_name: str) -> None:
    transport = MockTransport(HttpResponse(status, {"error": f"secret echo {FAKE_SECRET}"}))
    provider = (
        OllamaModelProvider(transport=transport)
        if provider_name == "ollama"
        else GroqModelProvider(transport=transport)
    )
    with pytest.raises(ProviderError, match=f"status {status}") as error:
        generate(provider)
    assert len(transport.calls) == 1
    assert FAKE_SECRET not in str(error.value)
    assert error.value.__context__ is None


@pytest.mark.parametrize("provider_name", ["ollama", "groq"])
@pytest.mark.parametrize("failure", [TimeoutError("secret echo"), RuntimeError(FAKE_SECRET)])
def test_transport_exception_has_no_raw_context_or_retry(
    provider_name: str, failure: Exception
) -> None:
    transport = MockTransport(failure)
    provider = (
        OllamaModelProvider(transport=transport)
        if provider_name == "ollama"
        else GroqModelProvider(transport=transport)
    )
    with pytest.raises(ProviderError, match="request failed") as error:
        generate(provider)
    assert len(transport.calls) == 1
    assert error.value.__context__ is error.value.__cause__ is None
    assert FAKE_SECRET not in str(error.value)


@pytest.mark.parametrize("provider_name", ["ollama", "groq"])
def test_all_returned_strings_are_redacted_with_captured_and_current_keys(
    provider_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider: OllamaModelProvider | GroqModelProvider
    updated_secret = "TEST_ONLY_ROTATED_CREDENTIAL_456"
    echoed = f"{FAKE_SECRET}:{updated_secret}"
    if provider_name == "ollama":
        body = ollama_body(
            model=echoed,
            message={"content": echoed, "thinking": echoed},
            done_reason=echoed,
            request_id=echoed,
        )
        provider = OllamaModelProvider(transport=MockTransport(HttpResponse(200, body)))
    else:
        body = groq_body(
            model=echoed,
            choices=[
                {"message": {"content": echoed, "reasoning": echoed}, "finish_reason": echoed}
            ],
            x_groq={"id": echoed},
        )
        provider = GroqModelProvider(transport=MockTransport(HttpResponse(200, body)))
    monkeypatch.setenv("GROQ_API_KEY", updated_secret)
    encoded = generate(provider).model_dump_json()
    assert FAKE_SECRET not in encoded and updated_secret not in encoded
    assert encoded.count("[REDACTED]") == 8


def test_public_redaction_helper_supports_cli_errors() -> None:
    assert redact(f"prefix {FAKE_SECRET} suffix") == "prefix [REDACTED] suffix"


@pytest.mark.parametrize("provider_name", ["ollama", "groq"])
def test_withdrawn_teacher_capability_blocks_adapter_before_transport(provider_name: str) -> None:
    transport = MockTransport(HttpResponse(200, {}))
    provider = (
        OllamaModelProvider(transport=transport)
        if provider_name == "ollama"
        else GroqModelProvider(transport=transport)
    )
    capability = TeacherCapability(provider, TeacherAccess.WITHDRAWN)
    with pytest.raises(TeacherUnavailable):
        asyncio.run(capability.generate(request()))
    assert transport.calls == []


@pytest.mark.parametrize(
    "configured",
    [
        "http://example.com:11434",
        "http://user:pass@localhost:11434",
        "http://localhost:11434/?key=secret",
        "http://localhost:11434/#secret",
        "http://localhost:11434/other",
        "http://localhost:0",
        "http://localhost:bad",
        "ftp://localhost:11434",
        "localhost:11434",
        "http://localhost:11434?",
        "http://localhost:11434#",
    ],
)
def test_ollama_host_is_loopback_and_never_accepts_url_credentials(
    configured: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", configured)
    with pytest.raises(ProviderError, match="loopback HTTP URL") as error:
        OllamaModelProvider(transport=MockTransport(HttpResponse(200, {})))
    assert configured not in str(error.value)
    assert error.value.__context__ is None


@pytest.mark.parametrize("configured", ["http://localhost:11434/", "http://[::1]:11434"])
def test_valid_loopback_host_is_used_without_extra_request(
    configured: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", configured)
    transport = MockTransport(HttpResponse(200, ollama_body()))
    generate(OllamaModelProvider(transport=transport))
    assert transport.calls[0][0] == configured.rstrip("/") + "/api/chat"
    assert len(transport.calls) == 1


def test_groq_credentials_are_environment_only_and_missing_key_fails_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(ProviderError, match="required in the environment"):
        GroqModelProvider(transport=MockTransport(HttpResponse(200, {})))


def test_provider_factory_selects_configured_provider_without_constructing_requests() -> None:
    assert isinstance(
        provider_factory(ModelConfig(provider="fake", model="fixture")), FakeModelProvider
    )
    assert isinstance(
        provider_factory(ModelConfig(provider="ollama", model="local")), OllamaModelProvider
    )
    assert isinstance(
        provider_factory(ModelConfig(provider="groq", model="remote")), GroqModelProvider
    )


class FakeHttpResponse:
    def __init__(self, status: int, content: bytes) -> None:
        self.status = status
        self.content = content
        self.reads = 0

    def getheader(self, name: str) -> str | None:
        return "request-header-1" if name == "x-request-id" else None

    def read(self) -> bytes:
        self.reads += 1
        return self.content


def test_stdlib_transport_uses_fresh_connections_and_exactly_one_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []

    class FakeConnection:
        def __init__(self, host: str, port: int | None, *, timeout: float) -> None:
            self.host, self.port, self.timeout = host, port, timeout
            self.requests: list[tuple[str, str, bytes, dict[str, str]]] = []
            self.closed = False
            instances.append(self)

        def request(self, method: str, path: str, *, body: bytes, headers: dict[str, str]) -> None:
            self.requests.append((method, path, body, headers))

        def getresponse(self) -> FakeHttpResponse:
            return FakeHttpResponse(200, b'{"ok":true}')

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("mosaic.providers.http.http.client.HTTPConnection", FakeConnection)
    transport = StdlibHttpTransport()
    for _ in range(2):
        response = asyncio.run(
            transport.post_json("http://localhost:11434/api/chat", {"a": 1}, {}, 5)
        )
        assert response.body == {"ok": True} and response.request_id == "request-header-1"
    assert len(instances) == 2
    for connection in instances:
        assert connection.closed and connection.timeout == 5
        assert len(connection.requests) == 1
        method, path, body, headers = connection.requests[0]
        assert method == "POST" and path == "/api/chat"
        assert json.loads(body) == {"a": 1} and headers["Content-Type"] == "application/json"


@pytest.mark.parametrize("status", [307, 401, 429])
def test_stdlib_transport_never_reads_redirect_or_error_body(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = FakeHttpResponse(status, FAKE_SECRET.encode())
    requests: list[str] = []

    class FakeConnection:
        def __init__(self, host: str, port: int | None, *, timeout: float) -> None:
            pass

        def request(self, method: str, path: str, *, body: bytes, headers: dict[str, str]) -> None:
            requests.append(method)

        def getresponse(self) -> FakeHttpResponse:
            return response

        def close(self) -> None:
            pass

    monkeypatch.setattr("mosaic.providers.http.http.client.HTTPConnection", FakeConnection)
    result = asyncio.run(StdlibHttpTransport().post_json("http://localhost/api/chat", {}, {}, 5))
    assert result.status == status and result.body == {}
    assert requests == ["POST"] and response.reads == 0
