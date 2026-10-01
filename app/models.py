"""Pydantic models for all tool inputs and outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Language = Literal["en", "hi", "mr"]

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}

Importance = int  # 1 = good to know, 2 = important, 3 = essential


# --- Resume -----------------------------------------------------------------


class Education(BaseModel):
    degree: str
    college: str = ""
    year: str | None = None  # kept as text: resumes say "2026", "2022-2026", "Expected 2026"


class Project(BaseModel):
    title: str
    tech: list[str] = Field(default_factory=list)
    description: str = ""


class ResumeProfile(BaseModel):
    name: str = ""
    email: str | None = None
    education: list[Education] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


# --- Role requirements ------------------------------------------------------


class Resource(BaseModel):
    title: str
    url: str
    type: str  # e.g. video, course, docs, article, practice


class SkillReq(BaseModel):
    name: str
    importance: Importance = Field(ge=1, le=3)
    category: str = ""
    aliases: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)  # skills that earn partial credit
    resources: list[Resource] = Field(default_factory=list)


class RoleRequirements(BaseModel):
    role: str
    family: str = ""
    summary: str = ""
    required_skills: list[SkillReq] = Field(default_factory=list)
    nice_to_have: list[SkillReq] = Field(default_factory=list)
    project_expectations: list[str] = Field(default_factory=list)


# --- Gap analysis -----------------------------------------------------------


class SkillJudgment(BaseModel):
    skill: str
    importance: Importance = Field(ge=1, le=3)
    status: Literal["matched", "partial", "missing"]
    reason: str = ""
    judged_by: Literal["rules", "llm"] = "rules"


class ScoreLine(BaseModel):
    points: float
    max_points: float
    label: str


class GapReport(BaseModel):
    role: str
    readiness_score: int = Field(ge=0, le=100)
    band: str
    matched: list[SkillJudgment] = Field(default_factory=list)
    partial: list[SkillJudgment] = Field(default_factory=list)
    missing: list[SkillJudgment] = Field(default_factory=list)
    score_breakdown: list[ScoreLine] = Field(default_factory=list)
    relevant_projects: list[str] = Field(default_factory=list)


# --- Roadmap ----------------------------------------------------------------


class RoadmapWeek(BaseModel):
    week: int = Field(ge=1)
    focus: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    resources: list[Resource] = Field(default_factory=list)
    mini_project: str = ""


class Roadmap(BaseModel):
    role: str
    weeks_available: int = Field(ge=1)
    language: Language = "en"
    weeks: list[RoadmapWeek] = Field(default_factory=list)
    later: list[str] = Field(default_factory=list)  # gaps that didn't fit in the available weeks


# --- Jobs and outreach ------------------------------------------------------


class JobMatch(BaseModel):
    id: str
    company: str
    role: str
    location: str = ""
    mode: str = ""  # e.g. remote, onsite, hybrid
    stipend: str | None = None
    match_percent: int = Field(ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    link: str = ""


class OutreachDraft(BaseModel):
    opening_id: str
    company: str
    channel: Literal["linkedin", "email"]
    subject: str | None = None
    message: str
    language: Language = "en"


# --- Agent run --------------------------------------------------------------


class TraceStep(BaseModel):
    step: int
    type: Literal["thought", "tool_call", "tool_result", "error", "final"]
    tool: str | None = None
    input: dict[str, Any] | None = None
    summary: str = ""


class AgentResult(BaseModel):
    """Everything one agent run produced. Every part is optional so a partial run still validates."""

    target_role: str
    language: Language = "en"
    llm_mode: str = "mock"
    profile: ResumeProfile | None = None
    requirements: RoleRequirements | None = None
    gap_report: GapReport | None = None
    roadmap: Roadmap | None = None
    matches: list[JobMatch] = Field(default_factory=list)
    outreach: OutreachDraft | None = None
    final_message: str = ""
    trace: list[TraceStep] = Field(default_factory=list)
