import hashlib
import platform
import subprocess
from importlib.metadata import distributions
from pathlib import Path

from mosaic.core.models import FrozenModel


class RunManifest(FrozenModel):
    mosaic_commit: str | None
    working_tree_dirty: bool | None
    source_sha256: str
    lockfile_sha256: str | None
    python_version: str
    platform: str
    dependencies: tuple[str, ...]
    environment_implementation: str
    provider_implementation: str


def _git(root: Path, *args: str) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(root), *args],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def manifest(environment: object, provider: object) -> RunManifest:
    root = Path(__file__).resolve().parents[2]
    status = _git(root, "status", "--porcelain")
    names = _git(root, "ls-files", "--cached", "--others", "--exclude-standard")
    if names is not None:
        files = names.splitlines()
    else:
        # Source archives have no Git metadata. Never hash installed .venv packages or traces.
        files = [
            str(path.relative_to(root))
            for directory in ("mosaic", "tests", "configs", "docs")
            for path in (root / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ]
        files.extend(
            [
                ".gitattributes",
                ".gitignore",
                ".python-version",
                "README.md",
                "pyproject.toml",
                "uv.lock",
            ]
        )
    digest = hashlib.sha256()
    for name in sorted(set(files)):
        path = root / name
        if path.is_file():
            digest.update(name.replace("\\", "/").encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    lock = root / "uv.lock"
    return RunManifest(
        mosaic_commit=_git(root, "rev-parse", "HEAD"),
        working_tree_dirty=None if status is None else bool(status),
        source_sha256=digest.hexdigest(),
        lockfile_sha256=hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
        python_version=platform.python_version(),
        platform=platform.platform(),
        dependencies=tuple(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())),
        environment_implementation=f"{type(environment).__module__}.{type(environment).__qualname__}",
        provider_implementation=f"{type(provider).__module__}.{type(provider).__qualname__}",
    )
