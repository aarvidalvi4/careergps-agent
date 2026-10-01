from pathlib import Path
from typing import Any

import pytest

from app.llm import BaseLLM, LLMError, LLMResponse, MockLLM
from app.models import GapReport, ResumeProfile
from app.tools.gap_analysis import analyze_gaps, llm_judge
from app.tools.resume_parser import parse_resume
from app.tools.role_lookup import lookup_role

SAMPLES = Path(__file__).parent / "samples"


def gaps(sample: str, role: str, llm: BaseLLM | None = None) -> GapReport:
    llm = llm or MockLLM()
    return analyze_gaps(parse_resume(SAMPLES / f"{sample}.pdf", MockLLM()), lookup_role(role), llm)


@pytest.fixture(scope="module")
def reports() -> dict[str, GapReport]:
    return {
        "strong": gaps("strong_data_analyst", "Data Analyst"),
        "medium": gaps("medium_data_analyst", "Data Analyst"),
        "weak": gaps("weak_ml_engineer", "ML Engineer"),
    }


def test_scores_are_ordered(reports: dict[str, GapReport]) -> None:
    assert reports["strong"].readiness_score > reports["medium"].readiness_score > reports["weak"].readiness_score


def test_score_is_deterministic(reports: dict[str, GapReport]) -> None:
    again = gaps("strong_data_analyst", "Data Analyst")
    assert again.readiness_score == reports["strong"].readiness_score
    assert again == reports["strong"]


@pytest.mark.parametrize("name", ["strong", "medium", "weak"])
def test_breakdown_is_consistent(reports: dict[str, GapReport], name: str) -> None:
    report = reports[name]
    assert abs(sum(line.points for line in report.score_breakdown) - report.readiness_score) <= 1
    assert all(line.points <= line.max_points for line in report.score_breakdown)
    assert abs(sum(line.max_points for line in report.score_breakdown) - 100) <= 1


def test_medium_has_partial_line(reports: dict[str, GapReport]) -> None:
    assert any("(partial," in line.label for line in reports["medium"].score_breakdown)
    assert reports["medium"].partial


def test_lists_sorted_by_importance(reports: dict[str, GapReport]) -> None:
    for report in reports.values():
        for group in (report.matched, report.partial, report.missing):
            assert [j.importance for j in group] == sorted((j.importance for j in group), reverse=True)


def test_bands(reports: dict[str, GapReport]) -> None:
    assert reports["strong"].band == "Ready to apply"
    assert reports["weak"].band == "Early stage"


# --- LLM judge ------------------------------------------------------------


class FakeLLM(BaseLLM):
    def __init__(self, result: Any = None, error: LLMError | None = None) -> None:
        self.result, self.error = result, error
        self.calls = 0

    def chat(self, messages, system=None, tools=None, max_tokens=2000) -> LLMResponse:
        raise AssertionError("not used")

    def complete_json(self, prompt: str, system: str | None = None) -> Any:
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def test_llm_judges_only_unresolved_in_one_call() -> None:
    llm = FakeLLM(
        {
            "judgments": [
                {"skill": "SQL", "status": "partial", "reason": "NPTEL course mentions databases"},
                {"skill": "Excel", "status": "missing", "reason": "should be ignored: already resolved"},
                {"skill": "Kubernetes", "status": "matched", "reason": "unknown skill, ignored"},
            ]
        }
    )
    report = gaps("medium_data_analyst", "Data Analyst", llm)
    assert llm.calls == 1
    sql = next(j for j in report.partial if j.skill == "SQL")
    assert sql.judged_by == "llm"
    assert any(j.skill == "Excel" and j.judged_by == "rules" for j in report.matched)


def test_llm_judge_validates_output() -> None:
    req = lookup_role("Data Analyst")
    skills = req.required_skills[:2]
    llm = FakeLLM({"judgments": [{"skill": skills[0].name, "status": "excellent"}, "junk", {"skill": skills[1].name.lower(), "status": "matched"}]})
    result = llm_judge(skills, ResumeProfile(), llm)
    assert list(result) == [skills[1].name]


def test_llm_error_falls_back_to_missing() -> None:
    report = gaps("medium_data_analyst", "Data Analyst", FakeLLM(error=LLMError("timeout")))
    sql = next(j for j in report.missing if j.skill == "SQL")
    assert sql.reason == "No evidence on the resume" and sql.judged_by == "rules"


def test_llm_not_called_when_mock_or_nothing_unresolved() -> None:
    assert llm_judge([], ResumeProfile(), FakeLLM({})) == {}
    assert llm_judge(lookup_role("Data Analyst").required_skills, ResumeProfile(), MockLLM()) == {}
