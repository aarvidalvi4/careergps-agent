"""The agent loop: the LLM decides which tool to call next; tools do the work and keep the results in AgentState."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from app.llm import BaseLLM, LLMError, Message, MockLLM, get_llm
from app.models import (
    LANGUAGE_NAMES,
    AgentResult,
    GapReport,
    JobMatch,
    Language,
    OutreachDraft,
    ResumeProfile,
    Roadmap,
    RoleRequirements,
    TraceStep,
)
from app.tools.gap_analysis import analyze_gaps
from app.tools.job_match import OpeningNotFound, get_opening, match_jobs
from app.tools.outreach import draft_outreach
from app.tools.resume_parser import parse_resume
from app.tools.roadmap import build_roadmap
from app.tools.role_lookup import RoleNotFound, list_roles, lookup_role
from app.tools.skills import load_openings

MAX_STEPS = 10


@dataclass
class AgentState:
    pdf_path: str | Path | BinaryIO
    target_role: str
    weeks: int = 8
    language: Language = "en"
    profile: ResumeProfile | None = None
    requirements: RoleRequirements | None = None
    gaps: GapReport | None = None
    roadmap: Roadmap | None = None
    matches: list[JobMatch] = field(default_factory=list)
    outreach: OutreachDraft | None = None


class ToolFailure(Exception):
    """A tool couldn't run with the given state/arguments. The message is sent back to the LLM so it can recover."""


# --- Tool wrappers ----------------------------------------------------------


def _int_arg(args: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(args.get(key, default))
    except (TypeError, ValueError):
        return default


def _need_profile(state: AgentState) -> ResumeProfile:
    if state.profile is None:
        raise ToolFailure("No resume parsed yet. Call parse_resume first.")
    return state.profile


def _need_role(state: AgentState) -> RoleRequirements:
    if state.requirements is None:
        raise ToolFailure("No role loaded yet. Call lookup_role first.")
    return state.requirements


def _need_gaps(state: AgentState) -> GapReport:
    if state.gaps is None:
        raise ToolFailure("No gap analysis yet. Call analyze_gaps first.")
    return state.gaps


def tool_parse_resume(state: AgentState, llm: BaseLLM, args: dict[str, Any]) -> dict[str, Any]:
    profile = parse_resume(state.pdf_path, llm)
    state.profile = profile
    return {
        "name": profile.name,
        "education": [e.degree for e in profile.education],
        "skills": profile.skills,
        "projects": [p.title for p in profile.projects],
        "experience_items": len(profile.experience),
        "certifications": profile.certifications,
    }


def tool_lookup_role(state: AgentState, llm: BaseLLM, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("role") or "").strip()
    if not query:
        raise ToolFailure(f"Missing 'role'. Supported roles: {', '.join(list_roles())}.")
    try:
        req = lookup_role(query)
    except RoleNotFound as exc:
        raise ToolFailure(
            f"Role {query!r} not found. Closest matches: {', '.join(exc.suggestions)}. "
            f"All supported roles: {', '.join(list_roles())}."
        ) from exc
    state.requirements = req
    return {
        "role": req.role,
        "family": req.family,
        "required_skills": [f"{s.name} (importance {s.importance})" for s in req.required_skills],
        "nice_to_have": [s.name for s in req.nice_to_have],
    }


def tool_analyze_gaps(state: AgentState, llm: BaseLLM, args: dict[str, Any]) -> dict[str, Any]:
    profile, req = _need_profile(state), _need_role(state)
    gaps = analyze_gaps(profile, req, llm)
    state.gaps = gaps
    return {
        "role": gaps.role,
        "score": gaps.readiness_score,
        "band": gaps.band,
        "matched": [j.skill for j in gaps.matched],
        "partial": [j.skill for j in gaps.partial],
        "missing_skills": [j.skill for j in gaps.missing],
        "relevant_projects": gaps.relevant_projects,
    }


def tool_build_roadmap(state: AgentState, llm: BaseLLM, args: dict[str, Any]) -> dict[str, Any]:
    gaps, req = _need_gaps(state), _need_role(state)
    roadmap = build_roadmap(gaps, req, llm, weeks=_int_arg(args, "weeks", state.weeks), language=state.language)
    state.roadmap = roadmap
    return {
        "weeks": [{"week": w.week, "focus": w.focus[0] if w.focus else ""} for w in roadmap.weeks],
        "did_not_fit": roadmap.later,
    }


def tool_match_jobs(state: AgentState, llm: BaseLLM, args: dict[str, Any]) -> dict[str, Any]:
    profile, gaps, req = _need_profile(state), _need_gaps(state), _need_role(state)
    limit = max(1, min(10, _int_arg(args, "limit", 5)))
    matches = match_jobs(profile, gaps, req, limit=limit)
    state.matches = matches
    return {
        "matches": [
            {"id": m.id, "company": m.company, "role": m.role, "match_percent": m.match_percent, "missing": m.missing_skills}
            for m in matches
        ]
    }


def tool_draft_outreach(state: AgentState, llm: BaseLLM, args: dict[str, Any]) -> dict[str, Any]:
    profile, gaps = _need_profile(state), _need_gaps(state)
    opening_id = str(args.get("opening_id") or "").strip()
    try:
        opening = get_opening(opening_id)
    except OpeningNotFound as exc:
        valid = [m.id for m in state.matches] or [o.id for o in load_openings()]
        raise ToolFailure(f"Unknown opening_id {opening_id!r}. Valid ids: {', '.join(valid)}.") from exc
    channel = args.get("channel", "linkedin")
    if channel not in ("linkedin", "email"):
        channel = "linkedin"
    draft = draft_outreach(profile, opening, gaps, llm, channel=channel, language=state.language)
    state.outreach = draft
    return {"opening_id": draft.opening_id, "company": draft.company, "channel": draft.channel, "subject": draft.subject, "message": draft.message}


# --- One-line summaries for the trace ----------------------------------------


def _summary_parse(state: AgentState, r: dict[str, Any]) -> str:
    return f"Read resume for {r['name'] or 'the student'}: {len(r['skills'])} skills, {len(r['projects'])} projects"


def _summary_role(state: AgentState, r: dict[str, Any]) -> str:
    return f"Loaded {r['role']}: {len(r['required_skills'])} required skills"


def _summary_gaps(state: AgentState, r: dict[str, Any]) -> str:
    return (
        f"Readiness {r['score']}/100 ({r['band']}). "
        f"{len(r['matched'])} matched, {len(r['partial'])} partial, {len(r['missing_skills'])} missing"
    )


def _summary_roadmap(state: AgentState, r: dict[str, Any]) -> str:
    later = f"; {len(r['did_not_fit'])} skill(s) saved for later" if r["did_not_fit"] else ""
    return f"Built a {len(r['weeks'])}-week roadmap{later}"


def _summary_jobs(state: AgentState, r: dict[str, Any]) -> str:
    if not r["matches"]:
        return "No openings found"
    top = r["matches"][0]
    return f"{len(r['matches'])} openings; best: {top['role']} at {top['company']} ({top['match_percent']}%)"


def _summary_outreach(state: AgentState, r: dict[str, Any]) -> str:
    channel = "email" if r["channel"] == "email" else "LinkedIn message"
    return f"Drafted a {channel} to {r['company']}"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[AgentState, BaseLLM, dict[str, Any]], dict[str, Any]]
    summarize: Callable[[AgentState, dict[str, Any]], str]

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}}

TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "parse_resume",
            "Read the student's uploaded resume PDF and extract name, education, skills, projects, experience and "
            "certifications. Call this first. Takes no arguments.",
            NO_ARGS,
            tool_parse_resume,
            _summary_parse,
        ),
        Tool(
            "lookup_role",
            "Load the requirements (required skills with importance 1-3, nice-to-haves) for a target job role. "
            "Accepts common names and spellings, e.g. 'data analyst intern' or 'machine learning engineer'.",
            {
                "type": "object",
                "properties": {"role": {"type": "string", "description": "The job role the student is targeting."}},
                "required": ["role"],
            },
            tool_lookup_role,
            _summary_role,
        ),
        Tool(
            "analyze_gaps",
            "Compare the parsed resume against the loaded role. Returns a 0-100 readiness score, a band, and which "
            "required skills are matched, partial or missing. Needs parse_resume and lookup_role first. No arguments.",
            NO_ARGS,
            tool_analyze_gaps,
            _summary_gaps,
        ),
        Tool(
            "build_roadmap",
            "Build a week-by-week learning plan that closes the gaps, one skill per week, ending with an application "
            "week. Needs analyze_gaps first.",
            {
                "type": "object",
                "properties": {"weeks": {"type": "integer", "description": "Weeks the student has (2-24).", "minimum": 2, "maximum": 24}},
                "required": ["weeks"],
            },
            tool_build_roadmap,
            _summary_roadmap,
        ),
        Tool(
            "match_jobs",
            "Rank sample internships and entry-level openings by how well the resume covers each one's skills. "
            "Returns opening ids, match percent and missing skills. Needs analyze_gaps first.",
            {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "How many openings to return (1-10).", "minimum": 1, "maximum": 10}},
            },
            tool_match_jobs,
            _summary_jobs,
        ),
        Tool(
            "draft_outreach",
            "Draft a short LinkedIn message or email to the company behind one opening, using an id from match_jobs.",
            {
                "type": "object",
                "properties": {
                    "opening_id": {"type": "string", "description": "An opening id returned by match_jobs, e.g. 'op01'."},
                    "channel": {"type": "string", "enum": ["linkedin", "email"], "description": "Defaults to linkedin."},
                },
                "required": ["opening_id"],
            },
            tool_draft_outreach,
            _summary_outreach,
        ),
    ]
}

SYSTEM_PROMPT = """You are CareerGPS, a career readiness agent for students at tier-2 and tier-3 colleges in India, many of whom have no placement mentor to tell them where they stand or what to do next.

You work by calling tools. Before EVERY tool call, write one short sentence saying what you are about to do and why. Put that sentence in the message text, never inside tool arguments. Tool arguments must be valid JSON matching the schema.

Typical flow:
1. parse_resume
2. lookup_role with the student's target role
3. analyze_gaps
4. build_roadmap with the weeks the student has
5. match_jobs
6. draft_outreach for the best-fitting opening. If the readiness score is very low (below 40), prefer an internship over a full-time role.

If a tool returns an ERROR, read it, fix the input (e.g. pick a suggested role, call a missing earlier step, use a valid opening id) and try again. Don't give up after one error.

When the work is done, reply WITHOUT any tool calls with 3-5 sentences written in the student's language: their score and what drives it, the single most important gap, and the first concrete action to take this week. Be honest and encouraging, never generic. Never invent numbers; only use what the tools returned."""


# --- Agent loop -------------------------------------------------------------


def _llm_mode(llm: BaseLLM) -> str:
    return "mock" if llm.is_mock else type(llm).__name__


def _fallback_final(state: AgentState) -> str:
    if state.gaps is None:
        return "I couldn't finish the analysis. Please check the resume and target role and try again."
    gaps = state.gaps
    top = gaps.missing[0].skill if gaps.missing else (gaps.partial[0].skill if gaps.partial else None)
    parts = [f"Your readiness for {gaps.role} is {gaps.readiness_score}/100 ({gaps.band})."]
    if top:
        parts.append(f"The most important gap is {top}; start on it this week using the first resource in your roadmap.")
    return " ".join(parts)


def _finish_with_rules(state: AgentState, step: Callable[..., dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Run whichever tools haven't completed yet, in the standard order, with the mock LLM (rule-based paths)."""
    rules = MockLLM()
    plan: list[tuple[str, Callable[[], bool], Callable[[], dict[str, Any]]]] = [
        ("parse_resume", lambda: state.profile is None, lambda: {}),
        ("lookup_role", lambda: state.requirements is None, lambda: {"role": state.target_role}),
        ("analyze_gaps", lambda: state.gaps is None, lambda: {}),
        ("build_roadmap", lambda: state.roadmap is None, lambda: {"weeks": state.weeks}),
        ("match_jobs", lambda: not state.matches, lambda: {"limit": 5}),
        (
            "draft_outreach",
            lambda: state.outreach is None and bool(state.matches),
            lambda: {"opening_id": state.matches[0].id, "channel": "linkedin"},
        ),
    ]
    for name, needed, make_args in plan:
        if not needed():
            continue
        args = make_args()
        tool = TOOLS[name]
        yield step("tool_call", f"Calling {name}", tool=name, input_=args)
        try:
            result = tool.run(state, rules, args)
            yield step("tool_result", tool.summarize(state, result), tool=name)
        except Exception as exc:
            yield step("error", str(exc), tool=name)
            return


def run_agent_events(
    pdf_path: str | Path | BinaryIO,
    target_role: str,
    weeks: int = 8,
    language: Language = "en",
    llm: BaseLLM | None = None,
) -> Iterator[dict[str, Any]]:
    """Run the agent, yielding {"event": "step", "data": TraceStep} as it goes and finally {"event": "result", ...}."""
    llm = llm or get_llm()
    state = AgentState(pdf_path=pdf_path, target_role=target_role, weeks=weeks, language=language)
    trace: list[TraceStep] = []
    final_message = ""

    def step(type_: str, summary: str, tool: str | None = None, input_: dict[str, Any] | None = None) -> dict[str, Any]:
        trace_step = TraceStep(step=len(trace) + 1, type=type_, tool=tool, input=input_, summary=summary)  # type: ignore[arg-type]
        trace.append(trace_step)
        return {"event": "step", "data": trace_step.model_dump()}

    messages: list[Message] = [
        {
            "role": "user",
            "content": (
                f"Target role: {target_role}\n"
                f"Weeks available: {weeks}\n"
                f"Language for the student: {LANGUAGE_NAMES[language]}\n"
                "The resume PDF is uploaded and ready for parse_resume."
            ),
        }
    ]
    schemas = [t.schema() for t in TOOLS.values()]

    for _ in range(MAX_STEPS):
        try:
            reply = llm.chat(messages, system=SYSTEM_PROMPT, tools=schemas)
        except LLMError as exc:
            yield step("error", f"The AI model failed: {exc}")
            # Degrade instead of dying: finish the remaining steps with the rule-based tools.
            yield step("thought", "The AI model is unavailable, so I'll finish the remaining steps with the rule-based tools.")
            yield from _finish_with_rules(state, step)
            final_message = _fallback_final(state)
            yield step("final", final_message)
            break

        if not reply.tool_calls:
            if state.gaps is not None and (state.roadmap is None or not state.matches or state.outreach is None):
                # The model stopped early; complete the skipped actions so the student gets the full result.
                yield from _finish_with_rules(state, step)
            final_message = reply.text.strip() or _fallback_final(state)
            yield step("final", final_message)
            break

        if reply.text.strip():
            yield step("thought", reply.text.strip())
        messages.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})

        for call in reply.tool_calls:
            yield step("tool_call", f"Calling {call.name}", tool=call.name, input_=call.arguments)
            tool = TOOLS.get(call.name)
            try:
                if tool is None:
                    raise ToolFailure(f"Unknown tool {call.name!r}. Available tools: {', '.join(TOOLS)}.")
                result = tool.run(state, llm, call.arguments or {})
                content = json.dumps(result, ensure_ascii=False)
                yield step("tool_result", tool.summarize(state, result), tool=call.name)
            except Exception as exc:  # any tool failure goes back to the LLM instead of crashing the run
                content = f"ERROR: {exc}"
                yield step("error", str(exc), tool=call.name)
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content})
    else:
        final_message = f"Stopped after {MAX_STEPS} steps without finishing. " + _fallback_final(state)
        yield step("final", final_message)

    result = AgentResult(
        target_role=target_role,
        language=language,
        llm_mode=_llm_mode(llm),
        profile=state.profile,
        requirements=state.requirements,
        gap_report=state.gaps,
        roadmap=state.roadmap,
        matches=state.matches,
        outreach=state.outreach,
        final_message=final_message,
        trace=trace,
    )
    yield {"event": "result", "data": result.model_dump()}


def run_agent(
    pdf_path: str | Path | BinaryIO,
    target_role: str,
    weeks: int = 8,
    language: Language = "en",
    llm: BaseLLM | None = None,
) -> AgentResult:
    result: dict[str, Any] | None = None
    for event in run_agent_events(pdf_path, target_role, weeks=weeks, language=language, llm=llm):
        if event["event"] == "result":
            result = event["data"]
    assert result is not None  # the generator always ends with a result event
    return AgentResult.model_validate(result)
