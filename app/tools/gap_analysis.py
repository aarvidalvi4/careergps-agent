"""Compare a resume against a role and compute a deterministic readiness score.

The LLM may only judge skills the rules couldn't resolve; every number is computed here.
"""

from __future__ import annotations

import json
from typing import Literal

from app.llm import BaseLLM, LLMError
from app.models import GapReport, ResumeProfile, RoleRequirements, ScoreLine, SkillJudgment, SkillReq
from app.tools.skills import find_terms_in_text, norm, skill_terms

SKILL_POINTS = 80.0
PROJECT_POINTS = {0: 0.0, 1: 7.0, 2: 13.0, 3: 20.0}  # 3+ relevant projects earn the full 20
PROJECT_MAX = 20.0
CREDIT = {"matched": 1.0, "partial": 0.5, "missing": 0.0}
Status = Literal["matched", "partial", "missing"]


def _terms_in(texts: list[str]) -> set[str]:
    found: set[str] = set()
    for text in texts:
        found.update(find_terms_in_text(text))
    return found


def _project_texts(profile: ResumeProfile) -> list[list[str]]:
    return [[p.title, p.description, *p.tech] for p in profile.projects]


def _profile_terms(profile: ResumeProfile) -> tuple[set[str], set[str]]:
    """(listed, used): terms from the skills list, and terms evidenced in projects/experience."""
    listed = {norm(s) for s in profile.skills} | _terms_in(profile.skills)
    used = _terms_in([t for texts in _project_texts(profile) for t in texts] + profile.experience)
    used |= {norm(t) for p in profile.projects for t in p.tech}
    return listed, used


def _skill_own_terms(skill: SkillReq) -> list[str]:
    return [norm(t) for t in [skill.name, *skill.aliases]]


def rule_judge(skill: SkillReq, listed: set[str], used: set[str]) -> SkillJudgment | None:
    def judgment(status: Status, reason: str) -> SkillJudgment:
        return SkillJudgment(skill=skill.name, importance=skill.importance, status=status, reason=reason, judged_by="rules")

    terms = _skill_own_terms(skill)
    for term in terms:
        if term in used:
            return judgment("matched", f"Used in projects/experience ({term})")
    for term in terms:
        if term in listed:
            return judgment("matched", f"Listed in skills ({term})")
    present = listed | used
    for related in skill.related:
        if any(t in present for t in skill_terms(related)):
            return judgment("partial", f"Has related skill ({related}), not {skill.name} itself")
    return None


LLM_SYSTEM = (
    "You are a strict technical recruiter checking a student's resume for specific skills. "
    "Judge only from the evidence given. Never assume."
)

LLM_PROMPT = """Decide for each skill below whether this student's resume shows it.

Statuses:
- "matched": clear evidence the student has used or listed this skill (or an equivalent tool).
- "partial": weak or indirect evidence, e.g. only a course or certificate name, or a closely related skill. A course name alone is partial at most.
- "missing": no real evidence.
In "reason", quote the exact resume evidence you relied on, or say "No evidence on the resume".

Skills to judge (name and tools that count as it):
{skills}

Student resume:
{resume}

Reply with JSON only: {{"judgments": [{{"skill": "<exact skill name from the list>", "status": "matched|partial|missing", "reason": "..."}}]}}"""


def llm_judge(unresolved: list[SkillReq], profile: ResumeProfile, llm: BaseLLM) -> dict[str, SkillJudgment]:
    """One batched LLM call for all skills the rules couldn't resolve. Returns {} on any failure."""
    if llm.is_mock or not unresolved:
        return {}
    skills_text = "\n".join(f"- {s.name}: {', '.join(s.aliases) or '(no aliases)'}" for s in unresolved)
    resume = json.dumps(
        {
            "skills": profile.skills,
            "projects": [p.model_dump() for p in profile.projects],
            "experience": profile.experience,
            "certifications": profile.certifications,
        },
        indent=1,
        ensure_ascii=False,
    )
    try:
        data = llm.complete_json(LLM_PROMPT.format(skills=skills_text, resume=resume), system=LLM_SYSTEM)
    except LLMError:
        return {}

    by_name = {s.name.lower(): s for s in unresolved}
    items = data.get("judgments", []) if isinstance(data, dict) else []
    results: dict[str, SkillJudgment] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        skill = by_name.get(str(item.get("skill", "")).strip().lower())
        status = str(item.get("status", "")).strip().lower()
        if skill is None or status not in CREDIT or skill.name in results:
            continue
        results[skill.name] = SkillJudgment(
            skill=skill.name,
            importance=skill.importance,
            status=status,  # type: ignore[arg-type]  # checked against CREDIT above
            reason=str(item.get("reason", "")).strip() or "Judged by the AI model",
            judged_by="llm",
        )
    return results


def _relevant_projects(profile: ResumeProfile, req: RoleRequirements) -> list[str]:
    role_terms = {t for skill in req.required_skills for t in _skill_own_terms(skill)}
    titles: list[str] = []
    for project, texts in zip(profile.projects, _project_texts(profile)):
        terms = _terms_in(texts) | {norm(t) for t in project.tech}
        if terms & role_terms:
            titles.append(project.title)
    return titles


def _band(score: int) -> str:
    if score < 40:
        return "Early stage"
    if score < 65:
        return "Building up"
    if score < 85:
        return "Nearly ready"
    return "Ready to apply"


def analyze_gaps(profile: ResumeProfile, req: RoleRequirements, llm: BaseLLM) -> GapReport:
    listed, used = _profile_terms(profile)

    judgments: dict[str, SkillJudgment] = {}
    unresolved: list[SkillReq] = []
    for skill in req.required_skills:
        result = rule_judge(skill, listed, used)
        if result:
            judgments[skill.name] = result
        else:
            unresolved.append(skill)

    judgments.update(llm_judge(unresolved, profile, llm))
    for skill in unresolved:
        judgments.setdefault(
            skill.name,
            SkillJudgment(skill=skill.name, importance=skill.importance, status="missing", reason="No evidence on the resume"),
        )

    by_importance = sorted(req.required_skills, key=lambda s: -s.importance)  # stable: keeps roles.json order on ties
    total_importance = sum(s.importance for s in req.required_skills) or 1

    breakdown: list[ScoreLine] = []
    total = 0.0
    for skill in by_importance:
        j = judgments[skill.name]
        max_points = SKILL_POINTS * skill.importance / total_importance
        points = max_points * CREDIT[j.status]
        total += points
        breakdown.append(
            ScoreLine(
                points=round(points, 1),
                max_points=round(max_points, 1),
                label=f"{skill.name} ({j.status}, importance {skill.importance}): {j.reason}",
            )
        )

    relevant = _relevant_projects(profile, req)
    project_points = PROJECT_POINTS[min(len(relevant), 3)]
    total += project_points
    breakdown.append(
        ScoreLine(
            points=project_points,
            max_points=PROJECT_MAX,
            label=f"Relevant projects: {len(relevant)} found (3+ earns full points)",
        )
    )

    score = round(total)
    ordered = [judgments[s.name] for s in by_importance]
    return GapReport(
        role=req.role,
        readiness_score=score,
        band=_band(score),
        matched=[j for j in ordered if j.status == "matched"],
        partial=[j for j in ordered if j.status == "partial"],
        missing=[j for j in ordered if j.status == "missing"],
        score_breakdown=breakdown,
        relevant_projects=relevant,
    )
