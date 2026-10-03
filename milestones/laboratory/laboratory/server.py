"""GET-only loopback server. It never imports the research runner or a provider."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlsplit

from laboratory.artifacts import ArtifactStore

STATIC = Path(__file__).parent / "static"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def make_server(
    store: ArtifactStore, port: int = 8765, allow_audit: bool = False
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def reply(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
            )
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def json_reply(self, status: int, value: object) -> None:
            self.reply(
                status,
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:
            actual_port = cast(ThreadingHTTPServer, self.server).server_port
            hosts = (f"127.0.0.1:{actual_port}", f"localhost:{actual_port}")
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if (
                host not in hosts
                or (origin is not None and origin != f"http://{host}")
                or self.headers.get("Sec-Fetch-Site") == "cross-site"
            ):
                self.json_reply(403, {"error": "Only the local viewer origin is allowed."})
                return
            target = urlsplit(self.path)
            parts = target.path.strip("/").split("/")
            query = parse_qs(target.query, keep_blank_values=True)
            try:
                if target.path in ASSETS and not target.query:
                    name, mime = ASSETS[target.path]
                    self.reply(200, (STATIC / name).read_bytes(), mime)
                elif target.path == "/api/runs" and not target.query:
                    self.json_reply(200, {"runs": store.catalog(), "audit_available": allow_audit})
                elif len(parts) == 3 and parts[:2] == ["api", "runs"] and not target.query:
                    self.json_reply(200, store.load(parts[2]))
                elif len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "audit":
                    if not allow_audit or query != {"ack": ["privileged-replay"]}:
                        self.json_reply(
                            403,
                            {
                                "error": "AUDIT VIEW requires server opt-in "
                                "and explicit replay acknowledgement."
                            },
                        )
                    else:
                        self.json_reply(200, store.audit(parts[2]))
                elif (
                    len(parts) == 5
                    and parts[:2] == ["api", "runs"]
                    and parts[3] == "frames"
                    and parts[4].isdigit()
                    and not target.query
                ):
                    raw, mime = store.frame(parts[2], int(parts[4]))
                    self.reply(200, raw, mime)
                else:
                    self.json_reply(404, {"error": "Unknown viewer resource."})
            except KeyError:
                self.json_reply(404, {"error": "Artifact unavailable."})
            except (OSError, ValueError, RecursionError):
                self.json_reply(422, {"error": "Artifact is invalid or exceeds viewer limits."})

        def reject_method(self) -> None:
            self.json_reply(405, {"error": "Read-only viewer: only GET is supported."})

        do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = reject_method

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
