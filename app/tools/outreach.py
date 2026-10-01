"""Draft a short LinkedIn message or email to a company about one opening."""

from __future__ import annotations

import json
from typing import Literal

from app.llm import BaseLLM, LLMError
from app.models import LANGUAGE_NAMES, GapReport, Language, Opening, OutreachDraft, Project, ResumeProfile
from app.tools.job_match import opening_skill_credits
from app.tools.skills import find_terms_in_text, skill_terms

Channel = Literal["linkedin", "email"]
WORD_LIMITS: dict[str, int] = {"linkedin": 90, "email": 150}


def _top_skills(profile: ResumeProfile, gaps: GapReport, opening: Opening, n: int = 3) -> list[str]:
    """The student's strongest skills for this opening: opening skills they have first, then other matches."""
    credits = opening_skill_credits(profile, gaps, opening)
    skills = [s for s, c in credits.items() if c == 1.0]
    skills += [j.skill for j in gaps.matched if j.skill not in skills]
    if not skills:
        skills = list(profile.skills)
    return skills[:n]


def _best_project(profile: ResumeProfile, gaps: GapReport, opening: Opening) -> Project | None:
    """The project that shows the most of the opening's skills (relevant projects win ties)."""
    if not profile.projects:
        return None
    wanted = {t for skill in opening.skills for t in skill_terms(skill)}

    def score(project: Project) -> tuple[int, int]:
        terms = set(find_terms_in_text(" ".join([project.title, project.description, *project.tech])))
        return len(terms & wanted), int(project.title in gaps.relevant_projects)

    return max(profile.projects, key=score)  # max keeps the first project on ties


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _fallback(profile: ResumeProfile, opening: Opening, skills: list[str], project: Project | None, channel: Channel) -> tuple[str | None, str]:
    name = profile.name or "a student"
    first_name = name.split()[0]
    degree = profile.education[0].degree if profile.education else ""
    intro = f"I'm {first_name} ({degree})" if degree else f"I'm {first_name}"
    skills_line = f" I work with {_join(skills)}." if skills else ""
    project_line = f" Most recently I worked on {project.title}." if project else ""

    if channel == "linkedin":
        message = (
            f"Hi {opening.company} team, {intro}, and I'm interested in the {opening.role} opening.{skills_line}{project_line} "
            "If it's useful, I'd be glad to share my resume. Would you be open to a short chat sometime?"
        )
        return None, message

    subject = f"{opening.role} application - {name}"
    project_detail = f" Most recently I worked on {project.title}: {project.description}" if project and project.description else project_line
    sign_off = f"{name}\n{profile.email}" if profile.email else name
    message = (
        f"Dear {opening.company} hiring team,\n\n"
        f"{intro}, and I'd like to apply for the {opening.role} role.{skills_line}{project_detail}\n\n"
        "I've attached my resume. If my background fits what you're looking for, I'd be happy to talk "
        "whenever convenient. Thank you for your time.\n\n"
        f"Regards,\n{sign_off}"
    )
    return subject, message


LLM_SYSTEM = "You help Indian students write short, honest outreach messages to employers."

LLM_PROMPT = """Write a {channel_name} to {company} about their "{role}" opening ({location}).

Rules:
- Under {limit} words.
- Name {company} and the {role} role in the message itself.
- Don't assume the student's gender from their name. In Hindi or Marathi, use phrasing that avoids gendered forms (e.g. "I'm Priya, B.Sc. Data Science" rather than words like छात्र/छात्रा).
- Specific and plain: no flattery, no buzzwords ("passionate", "synergy", "esteemed organisation", "go-getter").
- Mention exactly one concrete project from the student's details below, and what it did.
- Only use facts from the student's details; don't invent experience, numbers or skills.
- End with a low-pressure ask.
- Write in {language}. Keep skill, tool and company names in English.
{subject_rule}

Student details:
{student}

Reply with JSON only: {{"subject": {subject_shape}, "message": "..."}}"""


def _llm_draft(
    profile: ResumeProfile,
    opening: Opening,
    skills: list[str],
    project: Project | None,
    channel: Channel,
    language: Language,
    llm: BaseLLM,
) -> tuple[str | None, str] | None:
    student = json.dumps(
        {
            "name": profile.name,
            "email": profile.email,
            "education": [e.model_dump() for e in profile.education],
            "relevant_skills": skills,
            "project": project.model_dump() if project else None,
            "experience": profile.experience,
        },
        indent=1,
        ensure_ascii=False,
    )
    email = channel == "email"
    prompt = LLM_PROMPT.format(
        channel_name="short email" if email else "LinkedIn message",
        company=opening.company,
        role=opening.role,
        location=opening.location or "location not given",
        limit=WORD_LIMITS[channel],
        language=LANGUAGE_NAMES[language],
        subject_rule="- Give a clear subject line (under 10 words)." if email else "- No subject line (set subject to null).",
        student=student,
        subject_shape='"..."' if email else "null",
    )
    try:
        data = llm.complete_json(prompt, system=LLM_SYSTEM)
    except LLMError:
        return None
    if not isinstance(data, dict):
        return None
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return None
    subject = data.get("subject")
    subject = subject.strip() if email and isinstance(subject, str) and subject.strip() else None
    return subject, message.strip()


def draft_outreach(
    profile: ResumeProfile,
    opening: Opening,
    gaps: GapReport,
    llm: BaseLLM,
    channel: Channel = "linkedin",
    language: Language = "en",
) -> OutreachDraft:
    skills = _top_skills(profile, gaps, opening)
    project = _best_project(profile, gaps, opening)
    subject, message = _fallback(profile, opening, skills, project, channel)
    used_language: Language = "en"

    if not llm.is_mock:
        drafted = _llm_draft(profile, opening, skills, project, channel, language, llm)
        if drafted:
            llm_subject, message = drafted
            subject = llm_subject or subject  # an email always keeps a subject
            used_language = language

    return OutreachDraft(
        opening_id=opening.id,
        company=opening.company,
        channel=channel,
        subject=subject if channel == "email" else None,
        message=message,
        language=used_language,
    )
