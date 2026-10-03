"""Explicit public projections; raw artifact dictionaries never reach the browser."""

import re
from typing import Any

Json = dict[str, Any]
DIMENSIONS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "provider_total_tokens",
)
_SECRET = re.compile(
    r"AIza[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9_-]{10,}"
    r"|(?:GEMINI_API_KEY|GROQ_API_KEY|Authorization|api[_-]?key)\s*[:=]\s*[^\s,;]+",
    re.IGNORECASE,
)


def obj(value: Any) -> Json:
    return value if isinstance(value, dict) else {}


def safe_text(value: Any) -> str | None:
    return _SECRET.sub("[credential redacted]", value) if isinstance(value, str) else None


def pick(value: Any, keys: tuple[str, ...]) -> Json:
    result: Json = {}
    for key in keys:
        item = obj(value).get(key)
        if item is None or isinstance(item, (bool, int, float)):
            result[key] = item
        elif isinstance(item, str):
            result[key] = safe_text(item)
    return result


def actions(value: Any) -> list[Json]:
    return (
        [pick(item, ("name", "description")) for item in value] if isinstance(value, list) else []
    )


def observation(value: Any) -> Json:
    return pick(value, ("text",))


def usage(value: Any) -> Json:
    return pick(value, DIMENSIONS + ("provider_request_id", "measurement", "measurement_source"))


def budget(value: Any) -> Json:
    source = obj(value)
    keys = (
        DIMENSIONS
        + (
            "model_calls",
            "environment_actions",
            "wall_clock_ms",
            "synthetic_usage_calls",
        )
        + tuple(f"known_{name}" for name in DIMENSIONS)
        + (
            "unknown_input_calls",
            "unknown_output_calls",
            "unknown_reasoning_calls",
            "unknown_cached_input_calls",
            "unknown_provider_total_calls",
        )
    )
    result = pick(source, keys)
    for role in ("weak", "teacher"):
        if isinstance(source.get(role), dict):
            result[role] = pick(source[role], keys)
    return result


def summary(value: Any) -> Json:
    source = obj(value)
    result = pick(
        source,
        (
            "experiment_id",
            "episode_id",
            "status",
            "reason",
            "steps",
            "task",
            "total_model_calls",
            "teacher_call_fraction",
            "escalations",
            "blocked_escalations",
            "budget_complete",
        ),
    )
    result["budget"] = budget(source.get("budget"))
    # Final outcome is research metadata. Verifier evidence remains privileged.
    result["verification"] = pick(source.get("verification"), ("goal_id", "succeeded"))
    return result


def metadata(first: Any, arm: Any, final: Any) -> Json:
    config = obj(obj(obj(first).get("payload")).get("config"))
    source_arm, source_final = obj(arm), obj(final)
    result = pick(config, ("experiment_id", "seed", "task"))
    result["environment"] = pick(config.get("environment"), ("name", "revision"))
    role = "teacher" if config.get("agent_model_role") == "teacher" else "weak"
    result["role"] = role
    result["models"] = {
        role: pick(config.get("model"), ("provider", "model")),
    }
    teacher = obj(config.get("teacher"))
    if teacher:
        result["models"]["teacher"] = pick(teacher.get("model"), ("provider", "model"))
    for key in ("seed", "task", "family", "role", "provider", "model", "budget_complete"):
        if key in source_arm:
            result.update(pick(source_arm, (key,)))
    if source_arm.get("role") in ("weak", "teacher"):
        result["models"][source_arm["role"]] = pick(source_arm, ("provider", "model"))
    result["experiment_id"] = result.get("experiment_id") or safe_text(
        source_final.get("experiment_id")
    )
    result["task"] = result.get("task") or safe_text(source_final.get("task"))
    result["status"] = (
        safe_text(source_arm.get("status") or source_final.get("status")) or "trace only"
    )
    result["fixture_only"] = any(
        item.get("provider") == "fake" for item in result["models"].values()
    )
    return result


def public_event(value: Any, index: int) -> Json:
    source = obj(value)
    result = pick(
        source,
        (
            "schema_version",
            "sequence",
            "timestamp",
            "experiment_id",
            "episode_id",
            "step",
        ),
    )
    result["index"] = index
    payload = obj(source.get("payload"))
    kind = safe_text(source.get("event_type") or payload.get("kind")) or "Unknown"
    result["event_type"] = kind
    clean: Json = {}
    if kind == "ObservationReceived":
        clean = {
            "observation": observation(payload.get("observation")),
            "available_actions": actions(payload.get("available_actions")),
        }
    elif kind == "ModelCalled":
        request = obj(payload.get("request"))
        clean = pick(payload, ("call_index", "model_role", "provider"))
        clean["request"] = pick(request, ("model", "goal", "step"))
        clean["request"].update(
            {
                "observation": observation(request.get("observation")),
                "available_actions": actions(request.get("available_actions")),
            }
        )
    elif kind == "ModelResponded":
        response = obj(payload.get("response"))
        clean = pick(payload, ("call_index", "model_role", "latency_ms"))
        clean["response"] = pick(response, ("provider", "model", "text", "finish_reason"))
        clean["response"]["usage"] = usage(response.get("usage"))
    elif kind in ("ActionProposed", "ActionDispatched", "ActionExecuted"):
        clean = pick(payload, ("model_role",))
        clean["action"] = pick(payload.get("action"), ("name",))
        if kind == "ActionExecuted":
            clean["result"] = pick(
                payload.get("result"), ("accepted", "reason", "terminated", "truncated")
            )
            clean["result"]["observation"] = observation(
                obj(payload.get("result")).get("observation")
            )
    elif kind == "ActionRejected":
        clean = pick(payload, ("model_role", "proposal", "reason", "dispatched"))
        if isinstance(payload.get("result"), dict):
            clean["result"] = pick(
                payload["result"], ("accepted", "reason", "terminated", "truncated")
            )
            clean["result"]["observation"] = observation(payload["result"].get("observation"))
    elif kind in ("TaskSucceeded", "VerificationPerformed", "TaskFailed"):
        clean = pick(payload, ("reason",))
        clean["verification"] = pick(payload.get("verification"), ("goal_id", "succeeded"))
    elif kind == "ExperimentFinished":
        clean["summary"] = summary(payload.get("summary"))
    elif kind == "EpisodeFinished":
        clean["budget"] = budget(payload.get("budget"))
    elif kind in ("ModelCallFailed", "RunErrored"):
        clean = pick(
            payload,
            (
                "call_index",
                "model_role",
                "provider",
                "model",
                "latency_ms",
                "stage",
                "error_type",
            ),
        )
        # Arbitrary exception messages can contain credentials or private state.
        clean["detail"] = "Failure recorded; raw exception text is not served."
    elif kind in (
        "TeacherEscalated",
        "TeacherInvocationBlocked",
        "TeacherEscalationEligible",
        "WeakAttemptStarted",
        "WeakRecoveryStarted",
    ):
        clean = pick(
            payload,
            (
                "provider",
                "model",
                "access",
                "reason",
                "phase_steps",
                "recoveries_started",
                "consecutive_failures",
                "window_steps",
                "recovery_number",
            ),
        )
    # Snapshot, manifest, config, skill provenance and unknown payloads are omitted by default.
    result["payload"] = clean
    return result


def _audit_scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _audit_scrub(item)
            for key, item in value.items()
            if not re.search(r"secret|credential|authorization|api.?key|password", key, re.I)
        }
    if isinstance(value, list):
        return [_audit_scrub(item) for item in value]
    return safe_text(value) if isinstance(value, str) else value


def audit_event(value: Any, index: int) -> Json | None:
    import json

    source, privileged = obj(value), {}
    payload = obj(source.get("payload"))
    snapshot = obj(payload.get("snapshot"))
    if isinstance(snapshot.get("state_json"), str):
        try:
            privileged["snapshot"] = _audit_scrub(json.loads(snapshot["state_json"]))
        except (ValueError, RecursionError):
            privileged["snapshot"] = "Snapshot JSON unavailable."
    verification = payload.get("verification")
    if verification is None:
        verification = obj(payload.get("summary")).get("verification")
    if isinstance(verification, dict):
        privileged["verification"] = _audit_scrub(
            pick(verification, ("goal_id", "succeeded", "evidence"))
        )
    if not privileged:
        return None
    return {
        "index": index,
        "sequence": source.get("sequence"),
        "step": source.get("step"),
        "event_type": safe_text(source.get("event_type")),
        "privileged": privileged,
    }
