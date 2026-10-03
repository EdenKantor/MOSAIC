"""Bounded artifact discovery under operator-declared roots, never browser paths."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from laboratory.projection import Json, audit_event, metadata, obj, public_event, safe_text, summary

MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_LINE_BYTES = 32 * 1024 * 1024
MAX_EVENTS = 100_000
MAX_RUNS = 1000


def reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def contained(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("Artifact escapes the declared root")
    relative = path.absolute().relative_to(root.absolute())
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink() or current.is_junction():
            raise ValueError("Linked artifact paths are not served")
    return resolved


def json_file(path: Path, root: Path) -> Json:
    if not path.exists():
        return {}
    checked = contained(path, root)
    if not checked.is_file() or checked.stat().st_size > MAX_LINE_BYTES:
        raise ValueError("Unsupported artifact size")
    return obj(json.loads(checked.read_text(encoding="utf-8"), parse_constant=reject_constant))


@dataclass(frozen=True)
class Run:
    identifier: str
    root: Path
    directory: Path
    label: str


class ArtifactStore:
    def __init__(self, roots: list[Path]) -> None:
        if not roots:
            raise ValueError("At least one run root is required")
        self.roots = tuple(root.resolve(strict=True) for root in roots)
        if any(not root.is_dir() for root in self.roots):
            raise ValueError("Run roots must be existing directories")
        self.runs: dict[str, Run] = {}
        for root_index, root in enumerate(self.roots):
            for directory, dirs, files in os.walk(root, followlinks=False):
                base = Path(directory)
                depth = len(base.relative_to(root).parts)
                dirs[:] = (
                    sorted(
                        name
                        for name in dirs
                        if not name.startswith(".")
                        and not (base / name).is_symlink()
                        and not (base / name).is_junction()
                    )
                    if depth < 12
                    else []
                )
                if "events.jsonl" not in files:
                    continue
                try:
                    contained(base / "events.jsonl", root)
                except (OSError, ValueError):
                    continue
                relative = base.relative_to(root).as_posix()
                identifier = hashlib.sha256(f"{root_index}:{relative}".encode()).hexdigest()[:24]
                label = (
                    f"root-{root_index + 1}/{relative}"
                    if relative != "."
                    else f"root-{root_index + 1}"
                )
                self.runs[identifier] = Run(identifier, root, base, safe_text(label) or "Saved run")
                if len(self.runs) >= MAX_RUNS:
                    return

    def run(self, identifier: str) -> Run:
        if identifier not in self.runs:
            raise KeyError("Unknown run identifier")
        return self.runs[identifier]

    def _events(self, run: Run, *, first_only: bool = False) -> tuple[list[Json], list[str]]:
        path = contained(run.directory / "events.jsonl", run.root)
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Trace exceeds the viewer size limit")
        events, warnings = [], []
        with path.open("rb") as stream:
            while line := stream.readline(MAX_LINE_BYTES + 1):
                if len(line) > MAX_LINE_BYTES:
                    warnings.append("Replay stopped at an oversized event.")
                    break
                if not line.strip():
                    continue
                try:
                    event = json.loads(line, parse_constant=reject_constant)
                    if not isinstance(event, dict):
                        raise ValueError("An event must be an object")
                    events.append(event)
                except (ValueError, UnicodeError, RecursionError):
                    warnings.append(
                        "Replay stopped at an incomplete or invalid event; "
                        "earlier events remain inspectable."
                    )
                    break
                if first_only:
                    break
                if len(events) >= MAX_EVENTS:
                    warnings.append("Replay stopped at the event count limit.")
                    break
        return events, warnings

    def _optional(self, run: Run, name: str, warnings: list[str]) -> Json:
        try:
            return json_file(run.directory / name, run.root)
        except (OSError, ValueError, RecursionError):
            warnings.append(f"Invalid optional {name} omitted; trace replay remains available.")
            return {}

    def _metadata(self, run: Run, events: list[Json]) -> tuple[Json, Json, list[str]]:
        warnings: list[str] = []
        final = self._optional(run, "summary.json", warnings)
        arm = self._optional(run, "arm.json", warnings)
        if not final:
            for event in reversed(events):
                candidate = obj(obj(event.get("payload")).get("summary"))
                if candidate:
                    final = candidate
                    break
        meta = metadata(events[0] if events else {}, arm, final)
        meta.update({"id": run.identifier, "label": run.label})
        return meta, final, warnings

    def catalog(self) -> list[Json]:
        catalog: list[Json] = []
        for run in self.runs.values():
            try:
                events, _ = self._events(run, first_only=True)
                meta, _, _ = self._metadata(run, events)
                catalog.append(meta)
            except (OSError, ValueError, RecursionError):
                catalog.append(
                    {"id": run.identifier, "label": run.label, "status": "unreadable", "models": {}}
                )
        return catalog

    def load(self, identifier: str) -> Json:
        run = self.run(identifier)
        events, warnings = self._events(run)
        meta, final, companion_warnings = self._metadata(run, events)
        warnings.extend(companion_warnings)
        frames = self._frames(
            run,
            {event.get("sequence") for event in events if isinstance(event.get("sequence"), int)},
        )
        return {
            "metadata": meta,
            "summary": summary(final),
            "warnings": warnings,
            "events": [public_event(event, index) for index, event in enumerate(events)],
            "frames": [
                {"sequence": sequence, "url": f"/api/runs/{identifier}/frames/{sequence}"}
                for sequence in sorted(frames)
            ],
        }

    def audit(self, identifier: str) -> Json:
        run = self.run(identifier)
        events, warnings = self._events(run)
        entries = [
            entry
            for index, event in enumerate(events)
            if (entry := audit_event(event, index)) is not None
        ]
        final = self._optional(run, "summary.json", warnings)
        return {
            "label": "AUDIT VIEW — privileged replay evidence; never actor input",
            "events": entries,
            "warnings": warnings,
            "final_verification": audit_event(
                {"payload": {"verification": final.get("verification")}}, -1
            ),
        }

    def _frames(self, run: Run, sequences: set[Any]) -> dict[int, Path]:
        manifest = self._optional(run, "viewer-frames.json", [])
        result: dict[int, Path] = {}
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("frames"), list):
            return result
        for item in manifest["frames"]:
            frame = obj(item)
            sequence, name = frame.get("sequence"), frame.get("path")
            if (
                type(sequence) is not int
                or sequence not in sequences
                or not isinstance(name, str)
                or frame.get("source") != "alem_official_renderer"
                or frame.get("view") != "actor_visible"
            ):
                continue
            relative = PurePosixPath(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or ":" in name
                or relative.suffix.lower() not in (".png", ".jpg", ".jpeg")
            ):
                continue
            try:
                path = contained(run.directory.joinpath(*relative.parts), run.root)
                if (
                    path.is_relative_to(run.directory.resolve())
                    and path.is_file()
                    and path.stat().st_size <= 16 * 1024 * 1024
                ):
                    result[sequence] = path
            except (OSError, ValueError):
                continue
        return result

    def frame(self, identifier: str, sequence: int) -> tuple[bytes, str]:
        run = self.run(identifier)
        events, _ = self._events(run)
        frames = self._frames(
            run,
            {event.get("sequence") for event in events if isinstance(event.get("sequence"), int)},
        )
        if sequence not in frames:
            raise KeyError("Frame unavailable")
        raw = contained(frames[sequence], run.root).read_bytes()
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return raw, "image/png"
        if raw.startswith(b"\xff\xd8\xff"):
            return raw, "image/jpeg"
        raise ValueError("Unsupported image content")
