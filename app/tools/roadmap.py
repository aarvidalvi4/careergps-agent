"""Turn a gap report into a week-by-week plan. Scheduling is rule-based; the LLM may only rewrite the text."""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.llm import BaseLLM, LLMError
from app.models import LANGUAGE_NAMES, GapReport, Language, Roadmap, RoadmapWeek, RoleRequirements, SkillJudgment, SkillReq

MIN_WEEKS, MAX_WEEKS = 2, 24
MAX_RESOURCES = 3
# Output budget for the rewrite call. Reasoning models spend tokens before answering, and
# Hindi/Marathi text needs several tokens per word, so scale with the plan length.
LLM_TOKENS_BASE, LLM_TOKENS_PER_WEEK, LLM_TOKENS_CAP = 1500, 350, 8000
APPLY = "Apply and reach out"
PORTFOLIO = "Portfolio project"
INTERVIEW = "Interview prep"

# What a first mini-project looks like for each skill category.
CATEGORY_PROJECTS = {
    "databases": "Load a small public dataset into a database and write 10 {skill} queries that answer real questions about it.",
    "analytics": "Take one public dataset and use {skill} to produce a short analysis with 3 clear findings.",
    "math": "Answer 3 questions about a public dataset using {skill}, and explain each result in plain words.",
    "tools": "Redo a small everyday task (a budget, an attendance sheet, a to-do list) using {skill}, start to finish.",
    "programming": "Write a small program that solves one practical problem using {skill}, with a README explaining it.",
    "ml": "Train and evaluate one simple model on a public dataset using {skill}, and report its metrics honestly.",
    "frontend": "Build a single page that uses {skill}, deploy it for free (e.g. GitHub Pages) and share the link.",
    "backend": "Build a tiny API with two or three endpoints that uses {skill}, and document how to run it.",
    "devops": "Package one of your existing projects using {skill} so anyone can run it with a single command.",
    "systems": "Complete a set of hands-on exercises that use {skill} and write down the commands you learned.",
    "security": "Complete one beginner lab on {skill} and publish a short write-up of your method.",
    "business": "Apply {skill} to a familiar process (college admissions, a canteen, a fest) and document it in 1-2 pages.",
    "soft skills": "Prepare and record a 3-minute talk explaining one of your projects using {skill}.",
}
DEFAULT_PROJECT = "Build one small, complete exercise that uses {skill} and put it on GitHub with a README."


@dataclass
class _Slot:
    kind: str  # "skill" | "portfolio" | "interview" | "apply"
    judgment: SkillJudgment | None = None
    part: int = 1  # 1 or 2 for a skill's first/second week
    index: int = 0  # nth portfolio week


def _schedule(todo: list[SkillJudgment], available: int) -> tuple[list[_Slot], list[str]]:
    """Assign skills to weeks (one skill per week), then fill spare weeks."""
    lengths = [2 if j.status == "missing" and j.importance == 3 else 1 for j in todo]
    if sum(lengths) > available:
        lengths = [1] * len(todo)

    slots: list[_Slot] = []
    later: list[str] = []
    for judgment, length in zip(todo, lengths):
        if len(slots) + length <= available:
            slots.extend(_Slot("skill", judgment, part=p) for p in range(1, length + 1))
        else:
            later.append(judgment.skill)

    spare = available - len(slots)
    for i in range(spare):
        if spare >= 2 and i == spare - 1:
            slots.append(_Slot("interview"))
        else:
            slots.append(_Slot("portfolio", index=i))
    slots.append(_Slot("apply"))
    return slots, later


def _fallback_week(slot: _Slot, skill: SkillReq | None, req: RoleRequirements) -> tuple[list[str], str]:
    role = req.role
    if slot.kind == "skill" and slot.judgment:
        name = slot.judgment.skill
        if slot.part == 2:
            return (
                [
                    f"Use {name} on a real dataset or problem, not a tutorial example",
                    f"Extend last week's {name} project with at least two new features or findings",
                    "Push the updated project to GitHub with a README that explains what you did",
                ],
                f"Extend last week's project: apply {name} to a real problem or public dataset (e.g. from data.gov.in or Kaggle) and write up 3 results.",
            )
        project = CATEGORY_PROJECTS.get(skill.category if skill else "", DEFAULT_PROJECT).format(skill=name)
        if slot.judgment.status == "partial":
            return (
                [
                    f"Build on what you already know ({slot.judgment.reason.lower()}) and learn {name} properly",
                    f"Finish at least one full module or tutorial on {name}",
                    f"Add {name} to a project you have already built",
                ],
                project,
            )
        return (
            [
                f"Complete the first resource for {name} and take notes on the key ideas",
                f"Practise {name} for at least 45 minutes on 5 days this week",
                f"Finish a small exercise that uses {name} and save it on GitHub",
            ],
            project,
        )
    if slot.kind == "portfolio":
        expectations = req.project_expectations
        if slot.index < len(expectations):
            idea = expectations[slot.index]
        else:
            idea = f"Polish your strongest {role} project: fix rough edges, add tests or a demo video, and get feedback from a senior."
        return (
            [
                f"Combine the skills from earlier weeks into one portfolio project for a {role} role",
                "Publish it on GitHub with a clear README, screenshots and how to run it",
                "Add the project to your resume with one line on the result or impact",
            ],
            idea,
        )
    if slot.kind == "interview":
        return (
            [
                "Prepare a 2-minute walkthrough of each project: problem, approach, result",
                f"Practise answers to 20 common {role} interview questions, out loud",
                "Do 2 mock interviews with a friend, senior or online platform",
            ],
            f"Write a one-page {role} interview cheat sheet covering your projects and the key concepts from your plan.",
        )
    return (
        [
            f"Apply to at least 10 {role} openings or internships",
            "Send 5 personalised messages to alumni, seniors or recruiters on LinkedIn",
            "Update your resume and LinkedIn with the skills and projects from this plan",
        ],
        "Track every application in a spreadsheet: company, role, date applied, status and follow-up date.",
    )


LLM_SYSTEM = "You are a practical career mentor for students at tier-2 and tier-3 colleges in India."

LLM_PROMPT = """Rewrite the goals and mini-project for each week of this {weeks}-week plan to become a {role}.

Rules:
- Write in {language}. Keep skill and tool names (e.g. SQL, Power BI, Python) in English.
- 2-3 goals per week, each concrete and measurable (counts, deliverables, deadlines).
- One mini_project per week, a single sentence or two. Prefer Indian datasets and contexts (data.gov.in, IPL, Indian Railways, mandi prices, UPI, local businesses, college life).
- Do not change the week numbers or focus. A skill's second week must build on its first week.
- Don't use calendar dates; express deadlines as "by the end of the week", written in {language}.
- When a goal needs a course or tutorial, use the week's listed resources; don't recommend other courses.

Plan:
{plan}

Reply with JSON only: {{"weeks": [{{"week": 1, "goals": ["..."], "mini_project": "..."}}]}}"""


def _llm_rewrite(weeks: list[RoadmapWeek], req: RoleRequirements, llm: BaseLLM, language: Language) -> bool:
    """Rewrite goals/mini_project in place from ONE LLM call. Returns True if any LLM text was used."""
    if llm.is_mock:
        return False
    plan = json.dumps(
        [
            {
                "week": w.week,
                "focus": w.focus,
                "resources": [r.title for r in w.resources],
                "goals": w.goals,
                "mini_project": w.mini_project,
            }
            for w in weeks
        ],
        indent=1,
        ensure_ascii=False,
    )
    prompt = LLM_PROMPT.format(weeks=len(weeks), role=req.role, language=LANGUAGE_NAMES[language], plan=plan)
    try:
        max_tokens = min(LLM_TOKENS_CAP, LLM_TOKENS_BASE + LLM_TOKENS_PER_WEEK * len(weeks))
        data = llm.complete_json(prompt, system=LLM_SYSTEM, max_tokens=max_tokens)
    except LLMError:
        return False

    items = data.get("weeks") if isinstance(data, dict) else None
    by_week = {w.week: w for w in weeks}
    used = False
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("week"), int):
            continue
        week = by_week.get(item["week"])
        if week is None:
            continue
        goals = item.get("goals")
        if isinstance(goals, list):
            clean = [g.strip() for g in goals if isinstance(g, str) and g.strip()][:3]
            if clean:
                week.goals = clean
                used = True
        mini = item.get("mini_project")
        if isinstance(mini, str) and mini.strip():
            week.mini_project = mini.strip()
            used = True
    return used


def build_roadmap(
    gaps: GapReport,
    req: RoleRequirements,
    llm: BaseLLM,
    weeks: int = 8,
    language: Language = "en",
) -> Roadmap:
    weeks = max(MIN_WEEKS, min(MAX_WEEKS, weeks))
    todo = sorted(gaps.missing + gaps.partial, key=lambda j: (-j.importance, j.status != "missing"))
    slots, later = _schedule(todo, weeks - 1)

    skills = {s.name: s for s in req.required_skills + req.nice_to_have}
    plan: list[RoadmapWeek] = []
    for number, slot in enumerate(slots, start=1):
        skill = skills.get(slot.judgment.skill) if slot.judgment else None
        goals, mini_project = _fallback_week(slot, skill, req)
        if slot.kind == "skill" and slot.judgment:
            focus = [slot.judgment.skill]
        else:
            focus = [{"portfolio": PORTFOLIO, "interview": INTERVIEW, "apply": APPLY}[slot.kind]]
        plan.append(
            RoadmapWeek(
                week=number,
                focus=focus,
                goals=goals,
                resources=list(skill.resources[:MAX_RESOURCES]) if skill else [],
                mini_project=mini_project,
            )
        )

    used_llm = _llm_rewrite(plan, req, llm, language)
    return Roadmap(
        role=req.role,
        weeks_available=weeks,
        language=language if used_llm else "en",
        weeks=plan,
        later=later,
    )
