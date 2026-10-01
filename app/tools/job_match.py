"""Rank sample openings by how well the student's resume covers each opening's skills."""

from __future__ import annotations

from app.models import GapReport, JobMatch, Opening, ResumeProfile, RoleRequirements
from app.tools.gap_analysis import CREDIT, _profile_terms
from app.tools.skills import load_openings, skill_terms

FAMILY_BONUS = 15  # ranking boost (not shown in match_percent) for openings in the target role's family


class OpeningNotFound(Exception):
    pass


def get_opening(opening_id: str) -> Opening:
    for opening in load_openings():
        if opening.id == opening_id:
            return opening.model_copy(deep=True)
    raise OpeningNotFound(f"No opening with id {opening_id!r}")


def _skill_credit(skill: str, statuses: dict[str, str], present: set[str]) -> float:
    """Gap-report status if the skill was analysed for the target role, else a direct resume check."""
    if skill in statuses:
        return CREDIT[statuses[skill]]
    return 1.0 if any(term in present for term in skill_terms(skill)) else 0.0


def opening_skill_credits(profile: ResumeProfile, gaps: GapReport, opening: Opening) -> dict[str, float]:
    """Credit (1 / 0.5 / 0) the student earns for each of the opening's skills."""
    statuses = {j.skill: j.status for j in gaps.matched + gaps.partial + gaps.missing}
    listed, used = _profile_terms(profile)
    return {skill: _skill_credit(skill, statuses, listed | used) for skill in opening.skills}


def match_jobs(profile: ResumeProfile, gaps: GapReport, req: RoleRequirements, limit: int = 5) -> list[JobMatch]:
    ranked: list[tuple[float, int, str, JobMatch]] = []
    for opening in load_openings():
        if not opening.skills:
            continue
        credits = opening_skill_credits(profile, gaps, opening)
        percent = round(100 * sum(credits.values()) / len(credits))
        match = JobMatch(
            id=opening.id,
            company=opening.company,
            role=opening.role,
            location=opening.location,
            mode=opening.mode,
            stipend=opening.stipend,
            match_percent=percent,
            matched_skills=[s for s, c in credits.items() if c == 1.0],
            missing_skills=[s for s, c in credits.items() if c < 1.0],
            link=opening.link,
        )
        rank = percent + (FAMILY_BONUS if opening.family == req.family else 0)
        ranked.append((rank, percent, opening.id, match))

    # Highest rank first; ties broken by raw percent, then id, so results are stable.
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [match for *_, match in ranked[:limit]]
