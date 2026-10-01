from pathlib import Path
from typing import Any

import pytest

from app.llm import BaseLLM, LLMError, LLMResponse, MockLLM
from app.models import GapReport, RoleRequirements
from app.tools.gap_analysis import analyze_gaps
from app.tools.resume_parser import parse_resume
from app.tools.roadmap import APPLY, INTERVIEW, PORTFOLIO, build_roadmap
from app.tools.role_lookup import lookup_role

SAMPLES = Path(__file__).parent / "samples"


def report_for(sample: str, role: str) -> tuple[GapReport, RoleRequirements]:
    req = lookup_role(role)
    return analyze_gaps(parse_resume(SAMPLES / f"{sample}.pdf", MockLLM()), req, MockLLM()), req


@pytest.fixture(scope="module")
def weak() -> tuple[GapReport, RoleRequirements]:
    return report_for("weak_ml_engineer", "ML Engineer")


@pytest.mark.parametrize("weeks", [4, 8, 16])
def test_weak_resume_schedule(weak: tuple[GapReport, RoleRequirements], weeks: int) -> None:
    gaps, req = weak
    roadmap = build_roadmap(gaps, req, MockLLM(), weeks=weeks)
    assert len(roadmap.weeks) == weeks
    assert roadmap.weeks[-1].focus == [APPLY]
    assert all(len(w.focus) == 1 for w in roadmap.weeks)
    scheduled = {w.focus[0] for w in roadmap.weeks}
    for judgment in gaps.missing:
        assert judgment.skill in scheduled or judgment.skill in roadmap.later
    assert [w.week for w in roadmap.weeks] == list(range(1, weeks + 1))


def test_weeks_are_clamped(weak: tuple[GapReport, RoleRequirements]) -> None:
    gaps, req = weak
    assert len(build_roadmap(gaps, req, MockLLM(), weeks=1).weeks) == 2
    assert len(build_roadmap(gaps, req, MockLLM(), weeks=99).weeks) == 24


def test_two_week_skills_and_spare_weeks(weak: tuple[GapReport, RoleRequirements]) -> None:
    gaps, req = weak
    roadmap = build_roadmap(gaps, req, MockLLM(), weeks=16)
    focus = [w.focus[0] for w in roadmap.weeks]
    # 3 importance-3 missing skills x 2 weeks + 5 others = 11; 15 available -> 4 spare
    assert focus[:2] == ["Python", "Python"]
    assert roadmap.later == []
    assert focus[11:15] == [PORTFOLIO, PORTFOLIO, PORTFOLIO, INTERVIEW]
    first, second = roadmap.weeks[0], roadmap.weeks[1]
    assert first.goals != second.goals and first.mini_project != second.mini_project
    assert "last week" in second.mini_project.lower()
    assert len({w.mini_project for w in roadmap.weeks if w.focus == [PORTFOLIO]}) == 3


def test_tight_plan_gives_one_week_each(weak: tuple[GapReport, RoleRequirements]) -> None:
    gaps, req = weak
    roadmap = build_roadmap(gaps, req, MockLLM(), weeks=8)
    skill_weeks = [w.focus[0] for w in roadmap.weeks[:-1]]
    assert len(skill_weeks) == len(set(skill_weeks)) == 7
    assert len(roadmap.later) == 1


def test_missing_before_partial_and_resources() -> None:
    gaps, req = report_for("medium_data_analyst", "Data Analyst")
    roadmap = build_roadmap(gaps, req, MockLLM(), weeks=8)
    assert roadmap.weeks[0].focus == ["SQL"]  # the only missing skill, importance 3
    for week in roadmap.weeks:
        assert len(week.resources) <= 3
    assert roadmap.weeks[0].resources and roadmap.language == "en"


# --- LLM rewrite ----------------------------------------------------------


class FakeLLM(BaseLLM):
    def __init__(self, result: Any = None, error: LLMError | None = None) -> None:
        self.result, self.error = result, error
        self.calls = 0

    def chat(self, messages, system=None, tools=None, max_tokens=2000) -> LLMResponse:
        raise AssertionError("not used")

    def complete_json(self, prompt: str, system: str | None = None, max_tokens: int = 2000) -> Any:
        self.calls += 1
        self.max_tokens = max_tokens
        if self.error:
            raise self.error
        return self.result


def test_llm_text_is_merged_by_week(weak: tuple[GapReport, RoleRequirements]) -> None:
    gaps, req = weak
    baseline = build_roadmap(gaps, req, MockLLM(), weeks=4)
    llm = FakeLLM({"weeks": [{"week": 2, "goals": ["लक्ष्य एक", "लक्ष्य दो"], "mini_project": "IPL डेटा"}, {"week": 99, "goals": ["x"]}]})
    roadmap = build_roadmap(gaps, req, llm, weeks=4, language="hi")
    assert llm.calls == 1
    assert roadmap.language == "hi"
    assert llm.max_tokens > 2000  # budget scales with plan length
    assert roadmap.weeks[1].goals == ["लक्ष्य एक", "लक्ष्य दो"]
    assert roadmap.weeks[1].mini_project == "IPL डेटा"
    assert roadmap.weeks[0].goals == baseline.weeks[0].goals  # untouched weeks keep fallback text
    assert [w.focus for w in roadmap.weeks] == [w.focus for w in baseline.weeks]


def test_llm_failure_keeps_english(weak: tuple[GapReport, RoleRequirements]) -> None:
    gaps, req = weak
    for llm in (FakeLLM(error=LLMError("timeout")), FakeLLM({"weeks": "nope"}), MockLLM()):
        roadmap = build_roadmap(gaps, req, llm, weeks=4, language="mr")
        assert roadmap.language == "en"
