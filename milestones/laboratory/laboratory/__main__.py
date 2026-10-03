"""Launch the local Laboratory without inference or experiment execution."""

import argparse
from pathlib import Path

from laboratory.artifacts import ArtifactStore
from laboratory.server import make_server


def main() -> None:
    parser = argparse.ArgumentParser(description="MOSAIC Laboratory — saved artifact replay")
    parser.add_argument(
        "--run-root",
        action="append",
        type=Path,
        required=True,
        help="Existing artifact root to whitelist; repeat for multiple roots",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-audit",
        action="store_true",
        help="Enable a separate privileged replay endpoint; browser opt-in still required",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        store = ArtifactStore(args.run_root)
        server = make_server(store, args.port, args.allow_audit)
    except (OSError, ValueError):
        parser.error("run roots must be readable directories and the local port must be available")
    print(f"MOSAIC Laboratory: http://127.0.0.1:{args.port}")
    print(f"Saved runs: {len(store.runs)} | read-only | no model calls")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
