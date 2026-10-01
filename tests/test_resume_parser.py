from pathlib import Path
from typing import Any

import pytest

from app.llm import BaseLLM, LLMError, LLMResponse, MockLLM
from app.tools.resume_parser import ResumeParseError, heuristic_parse, parse_resume

SAMPLES = Path(__file__).parent / "samples"


def parse_sample(name: str):
    return parse_resume(SAMPLES / name, MockLLM())


def test_strong_resume() -> None:
    profile = parse_sample("strong_data_analyst.pdf")
    assert profile.name == "Priya Deshmukh"
    assert profile.email == "priya.deshmukh@example.com"
    lower = {s.lower() for s in profile.skills}
    assert {"sql", "python", "pandas", "power bi"} <= lower
    assert "Power BI" in profile.skills
    assert "testing" not in lower
    assert "excel" not in lower  # covered by "Advanced Excel"
    assert len(profile.projects) == 3
    assert profile.projects[0].title == "APMC Mandi Price Dashboard"
    assert profile.projects[0].tech == ["Power BI"]
    assert profile.projects[2].description.endswith("short report.")  # wrapped line joined
    assert len(profile.experience) == 1


def test_medium_resume() -> None:
    profile = parse_sample("medium_data_analyst.pdf")
    assert len(profile.projects) == 1
    assert profile.education[0].degree == "B.Com"
    assert profile.certifications == ["NPTEL: The Joy of Computing using Python"]


def test_weak_resume() -> None:
    profile = parse_sample("weak_ml_engineer.pdf")
    assert len(profile.projects) == 1
    assert "C" not in profile.skills


def test_fake_pdf_raises(tmp_path: Path) -> None:
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"%PDF-1.4 not really")
    with pytest.raises(ResumeParseError):
        parse_resume(fake, MockLLM())


def test_bullet_project_format() -> None:
    text = """Asha Rao
Projects
Movie Recommender
- Built a recommender with Python and scikit-learn
- Deployed with Flask
Expense Tracker
Tech: React, Node.js
- Tracks monthly spending
"""
    projects = heuristic_parse(text).projects
    assert [p.title for p in projects] == ["Movie Recommender", "Expense Tracker"]
    assert projects[0].description == "Built a recommender with Python and scikit-learn Deployed with Flask"
    assert projects[1].tech == ["React", "Node.js"]


# --- LLM path -------------------------------------------------------------


class FakeLLM(BaseLLM):
    def __init__(self, result: Any = None, error: LLMError | None = None) -> None:
        self.result, self.error = result, error

    def chat(self, messages, system=None, tools=None, max_tokens=2000) -> LLMResponse:
        raise AssertionError("not used")

    def complete_json(self, prompt: str, system: str | None = None) -> Any:
        if self.error:
            raise self.error
        return self.result


def test_llm_result_gets_missing_skills_added() -> None:
    llm = FakeLLM({"name": "Priya D", "skills": ["Python", "SQL (PostgreSQL)"], "projects": [{"title": "Dash"}]})
    profile = parse_resume(SAMPLES / "strong_data_analyst.pdf", llm)
    assert profile.name == "Priya D"
    assert "Power BI" in profile.skills  # added by the safety net
    assert "PostgreSQL" not in profile.skills  # already covered by "SQL (PostgreSQL)"
    assert profile.skills.count("Python") == 1


def test_llm_bad_json_falls_back_to_rules() -> None:
    profile = parse_resume(SAMPLES / "strong_data_analyst.pdf", FakeLLM(error=LLMError("not json")))
    assert profile.name == "Priya Deshmukh" and len(profile.projects) == 3


def test_llm_invalid_shape_falls_back_to_rules() -> None:
    profile = parse_resume(SAMPLES / "strong_data_analyst.pdf", FakeLLM(["not", "a", "profile"]))
    assert profile.name == "Priya Deshmukh"


def test_llm_auth_error_is_raised() -> None:
    with pytest.raises(LLMError):
        parse_resume(SAMPLES / "strong_data_analyst.pdf", FakeLLM(error=LLMError("bad key", status_code=401)))


def test_llm_rate_limit_falls_back() -> None:
    profile = parse_resume(SAMPLES / "strong_data_analyst.pdf", FakeLLM(error=LLMError("slow down", status_code=429)))
    assert profile.name == "Priya Deshmukh"
