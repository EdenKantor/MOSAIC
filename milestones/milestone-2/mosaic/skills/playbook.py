"""A single-writer local PlayBook; atomic replacement prevents partial document writes.

This file provides persistence and integrity checks, not an authentication boundary. A
trusted experiment supplies validation reports; this store cannot prove their truth.
Multiple processes must use separate stores or arrange their own writer coordination.
"""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from mosaic.core.models import FrozenModel, Observation
from mosaic.skills.models import (
    CandidateSkill,
    SkillScope,
    ValidationReport,
    VerifiedSkill,
    project_observation,
)


class _Document(FrozenModel):
    schema_version: Literal[1] = 1
    entries: tuple[VerifiedSkill, ...] = ()

    @model_validator(mode="after")
    def unique_entries(self) -> Self:
        keys = {(entry.candidate.skill_id, entry.candidate.version) for entry in self.entries}
        if len(keys) != len(self.entries):
            raise ValueError("PlayBook contains duplicate skill versions")
        return self


class PlayBook:
    def __init__(self, path: Path) -> None:
        self.path = path

    def entries(self) -> tuple[VerifiedSkill, ...]:
        try:
            data = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ()
        # Reject duplicate keys/non-finite values and require an explicit storage header.
        project_observation(data, ("schema_version", "entries"))
        document = _Document.model_validate_json(data)
        return tuple(
            sorted(
                document.entries,
                key=lambda entry: (entry.candidate.skill_id, entry.candidate.version),
            )
        )

    def promote(self, candidate: CandidateSkill, report: ValidationReport) -> bool:
        # Revalidate serialized input because Pydantic model_copy/construct can bypass validators.
        verified = VerifiedSkill.model_validate_json(
            VerifiedSkill(candidate=candidate, validation=report).model_dump_json()
        )
        entries = self.entries()
        for entry in entries:
            existing = entry.candidate
            if (existing.skill_id, existing.version) == (candidate.skill_id, candidate.version):
                if existing.content_sha256 != candidate.content_sha256:
                    raise ValueError("PlayBook skill identity collision with divergent content")
                # Preserve the first source/report when the same procedure is learned again.
                return False
        document = _Document(
            entries=tuple(
                sorted(
                    (*entries, verified),
                    key=lambda entry: (entry.candidate.skill_id, entry.candidate.version),
                )
            )
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self.path.parent,
                prefix=".playbook-",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(document.model_dump_json(indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return True

    def retrieve(self, scope: SkillScope, observation: Observation) -> VerifiedSkill | None:
        return next(
            (entry for entry in self.entries() if entry.candidate.matches(scope, observation)), None
        )

    def fingerprint(self) -> str | None:
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            return None
        return hashlib.sha256(data).hexdigest()
