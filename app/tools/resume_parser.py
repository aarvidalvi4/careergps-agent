"""Turn a resume PDF into a ResumeProfile, via the LLM when available and rules otherwise."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import BinaryIO

import pdfplumber
from pydantic import ValidationError

from app.llm import BaseLLM, LLMError
from app.models import Education, Project, ResumeProfile
from app.tools.skills import alias_map, find_terms_in_text, norm

MAX_PAGES = 5
MIN_TEXT_CHARS = 40
MAX_LLM_CHARS = 12000


class ResumeParseError(Exception):
    """The PDF could not be read, or holds no extractable text."""


# --- PDF text ---------------------------------------------------------------


def extract_text(pdf_path: str | Path | BinaryIO) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages[:MAX_PAGES])
    except Exception as exc:  # pdfplumber/pdfminer raise many different error types
        raise ResumeParseError(f"Could not read the PDF: {exc}") from exc
    if len(text.strip()) < MIN_TEXT_CHARS:
        raise ResumeParseError(
            "This PDF has almost no selectable text; it looks scanned. "
            "Please export your resume as a text PDF (e.g. 'Save as PDF' from Word or Google Docs) and upload again."
        )
    return text


# --- Rule-based parsing -----------------------------------------------------

SECTION_HEADINGS = {
    "skills": "skills",
    "technical skills": "skills",
    "projects": "projects",
    "academic projects": "projects",
    "experience": "experience",
    "work experience": "experience",
    "internships": "experience",
    "internship": "experience",
    "education": "education",
    "certifications": "certifications",
    "certificates": "certifications",
}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
BULLET_RE = re.compile(r"^\s*[-•*▪●◦]\s*")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}(?:\s*[-–]\s*(?:(?:19|20)\d{2}|present))?\b", re.IGNORECASE)
# "Title (tech, tech): description" on one line.
INLINE_PROJECT_RE = re.compile(r"^(?P<title>[^:()]{3,120}?)\s*(?:\((?P<tech>[^)]*)\))?\s*:\s*(?P<desc>.*)$")
TECH_LINE_RE = re.compile(r"^(?:tech(?:nologies)?|tech stack|stack|tools|built with)\s*:\s*(?P<tech>.+)$", re.IGNORECASE)


def _split_sections(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Return (header lines before the first heading, {section: lines})."""
    header: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        key = SECTION_HEADINGS.get(norm(line.rstrip(":")))
        if key:
            current = key
            sections.setdefault(key, [])
        elif current is None:
            header.append(line)
        else:
            sections[current].append(line)
    return header, sections


def _join_wrapped(lines: list[str]) -> list[str]:
    """Merge wrapped lines: a line starting lowercase continues the previous item."""
    items: list[str] = []
    for line in lines:
        is_bullet = bool(BULLET_RE.match(line))
        line = BULLET_RE.sub("", line)
        if items and not is_bullet and line[:1].islower():
            items[-1] = f"{items[-1]} {line}"
        else:
            items.append(line)
    return items


def _split_tech(raw: str) -> list[str]:
    return [t.strip() for t in re.split(r"[,/|]", raw) if t.strip()]


def _parse_education(lines: list[str]) -> list[Education]:
    entries: list[Education] = []
    for item in _join_wrapped(lines):
        year_match = YEAR_RE.search(item)
        parts = [p.strip() for p in item.split(",") if p.strip()]
        rest = [p for p in parts[1:] if not YEAR_RE.search(p) and not re.match(r"(?i)(cgpa|gpa|percentage|\d)", p)]
        entries.append(
            Education(degree=parts[0] if parts else item, college=", ".join(rest), year=year_match.group(0) if year_match else None)
        )
    return entries


def _parse_projects(lines: list[str]) -> list[Project]:
    projects: list[Project] = []
    descriptions: list[list[str]] = []
    last_kind = ""  # "title" | "inline" | "bullet"

    for line in lines:
        if BULLET_RE.match(line):
            text = BULLET_RE.sub("", line)
            if not projects:  # bullets before any title: treat the bullet as the title
                projects.append(Project(title=text))
                descriptions.append([])
                last_kind = "title"
            else:
                descriptions[-1].append(text)
                last_kind = "bullet"
            continue

        tech_line = TECH_LINE_RE.match(line)
        if tech_line and projects:
            projects[-1].tech.extend(_split_tech(tech_line.group("tech")))
            continue

        if ":" in line:
            inline = INLINE_PROJECT_RE.match(line)
            if inline:
                projects.append(Project(title=inline.group("title").strip(), tech=_split_tech(inline.group("tech") or "")))
                descriptions.append([inline.group("desc").strip()])
                last_kind = "inline"
                continue

        # A line with no colon right after an inline project, or a lowercase line after a
        # bullet, is a wrapped continuation of the description, not a new project.
        if projects and (last_kind == "inline" or (last_kind == "bullet" and line[:1].islower())):
            descriptions[-1].append(line)
            continue

        projects.append(Project(title=line))
        descriptions.append([])
        last_kind = "title"

    for project, parts in zip(projects, descriptions):
        project.description = " ".join(p for p in parts if p)
    return projects


def _term_regex(term: str) -> re.Pattern[str]:
    words = r"\s+".join(re.escape(w) for w in term.split())
    return re.compile(rf"(?<![A-Za-z0-9+#]){words}(?![A-Za-z0-9+#])(?!\.[A-Za-z0-9])", re.IGNORECASE)


def _label(term: str, text: str) -> str:
    """Readable label: the resume's own spelling unless it is all lowercase, else the canonical name."""
    match = _term_regex(term).search(text)
    if match and match.group(0) != match.group(0).lower():
        return " ".join(match.group(0).split())
    canonical = alias_map().get(term, "")
    return canonical if norm(canonical) == term else term


def extract_skills(text: str) -> list[str]:
    """Known skill terms in the text as readable labels, dropping terms contained in another found term."""
    terms = find_terms_in_text(text)
    kept = [t for t in terms if not any(t != other and f" {t} " in f" {other} " for other in terms)]
    labels: list[str] = []
    seen: set[str] = set()
    for term in kept:
        label = _label(term, text)
        if label.lower() not in seen:
            seen.add(label.lower())
            labels.append(label)
    return labels


def heuristic_parse(text: str) -> ResumeProfile:
    header, sections = _split_sections(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    email = EMAIL_RE.search(text)
    return ResumeProfile(
        name=lines[0] if lines else "",
        email=email.group(0) if email else None,
        education=_parse_education(sections.get("education", [])),
        skills=extract_skills(text),
        projects=_parse_projects(sections.get("projects", [])),
        experience=_join_wrapped(sections.get("experience", [])),
        certifications=_join_wrapped(sections.get("certifications", [])),
    )


# --- LLM parsing ------------------------------------------------------------

SYSTEM_PROMPT = "You extract structured data from student resumes. Only include what the resume says; never invent details."

PROMPT_TEMPLATE = """Extract this resume into JSON with exactly these keys:
{{
  "name": str,
  "email": str or null,
  "education": [{{"degree": str, "college": str, "year": str or null}}],
  "skills": [str],
  "projects": [{{"title": str, "tech": [str], "description": "one line"}}],
  "experience": [str],
  "certifications": [str]
}}
Only include what the resume says, don't invent. Use empty lists for missing sections.

RESUME:
{text}"""


def _is_config_error(exc: LLMError) -> bool:
    # 4xx means a bad key, model name or request; surface it. 429 (rate limit) is transient, so fall back.
    return exc.status_code is not None and 400 <= exc.status_code < 500 and exc.status_code != 429


def parse_resume(pdf_path: str | Path | BinaryIO, llm: BaseLLM) -> ResumeProfile:
    text = extract_text(pdf_path)
    if llm.is_mock:
        return heuristic_parse(text)

    try:
        data = llm.complete_json(PROMPT_TEMPLATE.format(text=text[:MAX_LLM_CHARS]), system=SYSTEM_PROMPT)
        profile = ResumeProfile.model_validate(data)
    except LLMError as exc:
        if _is_config_error(exc):
            raise
        return heuristic_parse(text)
    except ValidationError:
        return heuristic_parse(text)

    # Safety net: add known skill terms from the resume that the LLM left out.
    covered = {norm(s) for s in profile.skills} | set(find_terms_in_text(json.dumps(profile.skills)))
    for label in extract_skills(text):
        if norm(label) not in covered and not set(find_terms_in_text(label)) & covered:
            profile.skills.append(label)
    return profile
