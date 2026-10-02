from dataclasses import dataclass

from mosaic.core.models import Action, SkillHint
from mosaic.skills.models import CandidateSkill


@dataclass
class SkillSession:
    """Episode-local cursor; actions still require weak inference and environment acceptance."""

    candidate: CandidateSkill
    cursor: int = 0
    failed: bool = False

    @property
    def completed(self) -> bool:
        return not self.failed and self.cursor == len(self.candidate.procedure)

    def hint(self) -> SkillHint | None:
        if self.failed or self.completed:
            return None
        return SkillHint(
            skill_id=self.candidate.skill_id,
            version=self.candidate.version,
            content_sha256=self.candidate.content_sha256,
            procedure=self.candidate.procedure,
            cursor=self.cursor,
        )

    def advance(self, action: Action | None, accepted: bool) -> bool:
        if not accepted or action != self.candidate.procedure[self.cursor]:
            self.failed = True
            return False
        self.cursor += 1
        return True
