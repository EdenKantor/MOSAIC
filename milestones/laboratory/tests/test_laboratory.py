"""Information-boundary, path confinement and read-only transport tests."""

import hashlib
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from laboratory.artifacts import ArtifactStore, contained
from laboratory.projection import metadata, public_event
from laboratory.server import make_server


def event(kind: str, payload: dict[str, Any], sequence: int = 0) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "sequence": sequence,
        "step": sequence,
        "experiment_id": "test",
        "event_type": kind,
        "payload": {"kind": kind, **payload},
    }


class LaboratoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "runs"
        self.root.mkdir()
        self.directory = self.root / "episode"
        self.directory.mkdir()
        self.trace = self.directory / "events.jsonl"
        self.events = [
            event(
                "ExperimentStarted",
                {
                    "config": {
                        "experiment_id": "fixture",
                        "seed": 42,
                        "task": "collect_wood",
                        "model": {
                            "provider": "fake",
                            "model": "fixture-policy",
                            "secret": "config-secret",
                        },
                        "environment": {"name": "alem", "revision": "pinned"},
                        "private_config": "must-not-appear",
                    },
                    "manifest": {"local_path": "private-machine-path"},
                },
            ),
            event(
                "EpisodeStarted",
                {
                    "snapshot": {
                        "state_json": json.dumps(
                            {"hidden_map": [7, 8], "rng": [3, 4], "api_key": "removed"}
                        )
                    }
                },
                1,
            ),
            event(
                "ObservationReceived",
                {
                    "observation": {
                        "text": "Visible tree. Inventory: wood 0",
                        "hidden_state": "nope",
                    },
                    "available_actions": [
                        {"name": "DO", "description": "Use facing tile", "private": "nope"}
                    ],
                    "private": "nope",
                },
                2,
            ),
            event(
                "ModelCalled",
                {
                    "call_index": 0,
                    "model_role": "weak",
                    "provider": "fake",
                    "request": {
                        "model": "fixture-policy",
                        "observation": {"text": "Visible tree"},
                        "available_actions": [{"name": "DO", "description": "Use facing tile"}],
                        "system_prompt": "private-prompt-field",
                        "snapshot": "nope",
                        "history": [{"provenance": "hidden-skill-source"}],
                    },
                },
                3,
            ),
            event(
                "ModelResponded",
                {
                    "call_index": 0,
                    "response": {
                        "provider": "fake",
                        "model": "fixture-policy",
                        "text": '{"name":"DO"}',
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 4,
                            "measurement": "synthetic",
                            "private": "nope",
                        },
                        "hidden_thoughts": "must-not-appear",
                    },
                    "latency_ms": 2,
                },
                4,
            ),
            event(
                "TaskSucceeded",
                {
                    "verification": {
                        "goal_id": "collect_wood",
                        "succeeded": True,
                        "evidence": "private-verifier-evidence",
                    }
                },
                5,
            ),
        ]
        self.write_events(self.events)
        (self.directory / "summary.json").write_text(
            json.dumps(
                {
                    "experiment_id": "fixture",
                    "status": "succeeded",
                    "task": "collect_wood",
                    "steps": 1,
                    "verification": {
                        "goal_id": "collect_wood",
                        "succeeded": True,
                        "evidence": "private-verifier-evidence",
                    },
                    "budget": {
                        "model_calls": 1,
                        "input_tokens": 12,
                        "output_tokens": 4,
                        "private": "nope",
                    },
                }
            ),
            encoding="utf-8",
        )

    def write_events(self, events: list[dict[str, Any]]) -> None:
        self.trace.write_text("".join(json.dumps(item) + "\n" for item in events), encoding="utf-8")

    def store(self) -> tuple[ArtifactStore, str]:
        store = ArtifactStore([self.root])
        return store, next(iter(store.runs))

    def test_public_projection_removes_private_nested_fields(self) -> None:
        store, identifier = self.store()
        public = json.dumps(store.load(identifier))
        for forbidden in (
            "hidden_map",
            "rng",
            "private-verifier-evidence",
            "local_path",
            "private_config",
            "config-secret",
            "private-prompt-field",
            "hidden-skill-source",
            "hidden_thoughts",
            "nope",
        ):
            self.assertNotIn(forbidden, public)
        self.assertIn("Visible tree", public)
        self.assertIn("fixture-policy", public)
        self.assertIn('"succeeded": true', public)

    def test_audit_contains_only_explicit_privileged_evidence(self) -> None:
        store, identifier = self.store()
        audit = json.dumps(store.audit(identifier))
        self.assertIn("AUDIT VIEW", audit)
        self.assertIn("hidden_map", audit)
        self.assertIn("private-verifier-evidence", audit)
        self.assertNotIn("api_key", audit)
        self.assertNotIn("private_config", audit)
        self.assertNotIn("local_path", audit)

    def test_unknown_payload_is_not_exposed(self) -> None:
        projected = public_event(event("FutureEvent", {"private": {"snapshot": "hidden"}}), 0)
        self.assertEqual(projected["payload"], {})

    def test_missing_legacy_usage_stays_unknown(self) -> None:
        projected = public_event(self.events[4], 4)
        usage = projected["payload"]["response"]["usage"]
        self.assertEqual(usage["input_tokens"], 12)
        for dimension in ("reasoning_tokens", "cached_input_tokens", "provider_total_tokens"):
            self.assertIsNone(usage[dimension])

    def test_provider_failure_message_is_never_served(self) -> None:
        projected = public_event(
            event(
                "ModelCallFailed",
                {
                    "model": "test",
                    "message": "exception: private-world + secret-value",
                    "error_type": "ProviderError",
                },
            ),
            0,
        )
        self.assertNotIn("secret-value", json.dumps(projected))
        self.assertEqual(projected["payload"]["error_type"], "ProviderError")

    def test_known_credential_patterns_redacted_without_environment_reads(self) -> None:
        with patch("os.getenv", side_effect=AssertionError("Viewer must not read credentials")):
            projected = public_event(
                event(
                    "ObservationReceived",
                    {"observation": {"text": "GROQ_API_KEY=fictional-secret gsk_abcdefghijklmnop"}},
                ),
                0,
            )
        self.assertNotIn("fictional-secret", json.dumps(projected))
        self.assertNotIn("gsk_", json.dumps(projected))

    def test_metadata_is_not_full_config(self) -> None:
        meta = metadata(self.events[0], {}, {})
        self.assertEqual(meta["seed"], 42)
        self.assertTrue(meta["fixture_only"])
        self.assertEqual(set(meta["models"]["weak"]), {"provider", "model"})
        self.assertNotIn("private_config", meta)

    def test_partial_trace_preserves_prior_complete_events(self) -> None:
        with self.trace.open("a", encoding="utf-8") as stream:
            stream.write('{"incomplete":')
        store, identifier = self.store()
        loaded = store.load(identifier)
        self.assertEqual(len(loaded["events"]), 6)
        self.assertTrue(loaded["warnings"])

    def test_oversized_artifact_rejected(self) -> None:
        store, identifier = self.store()
        with patch("laboratory.artifacts.MAX_FILE_BYTES", 1), self.assertRaises(ValueError):
            store.load(identifier)

    def test_invalid_optional_summary_does_not_hide_trace(self) -> None:
        (self.directory / "summary.json").write_text('{"unfinished":', encoding="utf-8")
        store, identifier = self.store()
        loaded = store.load(identifier)
        self.assertEqual(len(loaded["events"]), 6)
        self.assertIn("summary.json", loaded["warnings"][0])

    def test_nonfinite_json_numbers_do_not_reach_browser(self) -> None:
        with self.trace.open("a", encoding="utf-8") as stream:
            stream.write('{"step": NaN}\n')
        store, identifier = self.store()
        loaded = store.load(identifier)
        self.assertEqual(len(loaded["events"]), 6)
        self.assertTrue(loaded["warnings"])

    def test_root_and_ids_do_not_admit_browser_paths(self) -> None:
        store, _ = self.store()
        for identifier in ("../episode", str(self.trace), "%2e%2e", "episode/events.jsonl"):
            with self.assertRaises(KeyError):
                store.load(identifier)
        outside = Path(self.temp.name) / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            contained(outside, self.root)

    def test_linked_artifact_is_rejected(self) -> None:
        with patch.object(Path, "is_symlink", return_value=True), self.assertRaises(ValueError):
            contained(self.trace, self.root)

    def test_frames_require_public_official_manifest_and_confined_path(self) -> None:
        frame = self.directory / "frame.png"
        frame.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        entries = [
            {
                "sequence": 2,
                "path": "frame.png",
                "source": "alem_official_renderer",
                "view": "actor_visible",
            },
            {
                "sequence": 3,
                "path": "../episode/frame.png",
                "source": "alem_official_renderer",
                "view": "actor_visible",
            },
            {
                "sequence": 4,
                "path": "frame.png",
                "source": "alem_official_renderer",
                "view": "full_world",
            },
            {
                "sequence": 5,
                "path": "frame.png",
                "source": "custom_renderer",
                "view": "actor_visible",
            },
        ]
        (self.directory / "viewer-frames.json").write_text(
            json.dumps({"schema_version": 1, "frames": entries}), encoding="utf-8"
        )
        store, identifier = self.store()
        self.assertEqual([f["sequence"] for f in store.load(identifier)["frames"]], [2])
        self.assertEqual(store.frame(identifier, 2)[1], "image/png")
        for sequence in (3, 4, 5, 999):
            with self.assertRaises(KeyError):
                store.frame(identifier, sequence)

    def test_server_is_get_only_and_does_not_modify_artifacts(self) -> None:
        store, identifier = self.store()
        before = hashlib.sha256(self.trace.read_bytes()).hexdigest()
        server = make_server(store, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]

            def request(
                method: str, target: str, headers: dict[str, str] | None = None
            ) -> tuple[int, bytes]:
                connection = http.client.HTTPConnection("127.0.0.1", port)
                connection.request(method, target, headers=headers or {})
                response = connection.getresponse()
                result = response.status, response.read()
                connection.close()
                return result

            self.assertEqual(request("GET", "/api/runs")[0], 200)
            status, body = request("GET", f"/api/runs/{identifier}")
            self.assertEqual(status, 200)
            self.assertNotIn(b"hidden_map", body)
            for method in ("POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
                self.assertEqual(request(method, "/api/runs")[0], 405)
            self.assertEqual(
                request("GET", f"/api/runs/{identifier}/audit?ack=privileged-replay")[0], 403
            )
            self.assertEqual(request("GET", "/api/runs", {"Host": "attacker.example"})[0], 403)
            self.assertEqual(
                request("GET", "/api/runs", {"Origin": "https://attacker.example"})[0], 403
            )
            self.assertEqual(request("GET", "/api/runs", {"Sec-Fetch-Site": "cross-site"})[0], 403)
            for target in (
                "/../summary.json",
                "/api/runs/..%2fepisode",
                "/api/runs?root=C:/",
                "/config.json",
            ):
                self.assertEqual(request("GET", target)[0], 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(hashlib.sha256(self.trace.read_bytes()).hexdigest(), before)

    def test_audit_server_requires_both_opt_ins(self) -> None:
        store, identifier = self.store()
        server = make_server(store, port=0, allow_audit=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for query, expected in (
                ("", 403),
                ("?ack=privileged-replay", 200),
                ("?ack=privileged-replay&extra=1", 403),
            ):
                connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
                connection.request("GET", f"/api/runs/{identifier}/audit{query}")
                response = connection.getresponse()
                self.assertEqual(response.status, expected)
                body = response.read()
                if expected == 200:
                    self.assertIn(b"hidden_map", body)
                connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
