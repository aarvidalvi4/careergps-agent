from pathlib import Path
from typing import Any

import pytest

from app.llm import BaseLLM, LLMError, LLMResponse, MockLLM
from app.models import GapReport, ResumeProfile
from app.tools.gap_analysis import analyze_gaps
from app.tools.job_match import get_opening
from app.tools.outreach import draft_outreach
from app.tools.resume_parser import parse_resume
from app.tools.role_lookup import lookup_role

SAMPLES = Path(__file__).parent / "samples"


@pytest.fixture(scope="module")
def strong() -> tuple[ResumeProfile, GapReport]:
    profile = parse_resume(SAMPLES / "strong_data_analyst.pdf", MockLLM())
    return profile, analyze_gaps(profile, lookup_role("Data Analyst"), MockLLM())


def test_email_has_subject_and_company(strong: tuple[ResumeProfile, GapReport]) -> None:
    profile, gaps = strong
    draft = draft_outreach(profile, get_opening("op02"), gaps, MockLLM(), channel="email")
    assert draft.subject
    assert "PaisaPath Fintech" in draft.message
    assert draft.channel == "email" and draft.opening_id == "op02" and draft.language == "en"
    assert len(draft.message.split()) <= 150


def test_linkedin_fallback(strong: tuple[ResumeProfile, GapReport]) -> None:
    profile, gaps = strong
    draft = draft_outreach(profile, get_opening("op02"), gaps, MockLLM())
    assert draft.subject is None
    assert len(draft.message.split()) < 90
    assert "Junior Data Analyst" in draft.message
    assert "SQL" in draft.message
    assert "Food-Delivery Orders SQL Case Study" in draft.message  # project closest to the opening's skills
    assert draft.message.rstrip().endswith("?")  # low-pressure ask


def test_handles_profile_without_projects_or_name() -> None:
    gaps = analyze_gaps(ResumeProfile(), lookup_role("Data Analyst"), MockLLM())
    draft = draft_outreach(ResumeProfile(), get_opening("op01"), gaps, MockLLM(), channel="email")
    assert "Kisanlytics" in draft.message and draft.subject


class FakeLLM(BaseLLM):
    def __init__(self, result: Any = None, error: LLMError | None = None) -> None:
        self.result, self.error = result, error

    def chat(self, messages, system=None, tools=None, max_tokens=2000) -> LLMResponse:
        raise AssertionError("not used")

    def complete_json(self, prompt: str, system: str | None = None, max_tokens: int = 2000) -> Any:
        if self.error:
            raise self.error
        return self.result


def test_llm_text_used_in_requested_language(strong: tuple[ResumeProfile, GapReport]) -> None:
    profile, gaps = strong
    llm = FakeLLM({"subject": "Junior Data Analyst पद के लिए आवेदन", "message": "नमस्ते PaisaPath Fintech टीम"})
    draft = draft_outreach(profile, get_opening("op02"), gaps, llm, channel="email", language="hi")
    assert draft.message == "नमस्ते PaisaPath Fintech टीम"
    assert draft.subject == "Junior Data Analyst पद के लिए आवेदन"
    assert draft.language == "hi"


def test_llm_email_without_subject_keeps_fallback_subject(strong: tuple[ResumeProfile, GapReport]) -> None:
    profile, gaps = strong
    draft = draft_outreach(profile, get_opening("op02"), gaps, FakeLLM({"message": "Hello"}), channel="email")
    assert draft.message == "Hello" and draft.subject


def test_llm_linkedin_subject_dropped(strong: tuple[ResumeProfile, GapReport]) -> None:
    profile, gaps = strong
    draft = draft_outreach(profile, get_opening("op02"), gaps, FakeLLM({"subject": "x", "message": "Hello"}))
    assert draft.subject is None


@pytest.mark.parametrize("llm", [FakeLLM(error=LLMError("timeout")), FakeLLM({"message": ""}), FakeLLM(["junk"])])
def test_llm_failure_falls_back_to_english(strong: tuple[ResumeProfile, GapReport], llm: BaseLLM) -> None:
    profile, gaps = strong
    draft = draft_outreach(profile, get_opening("op02"), gaps, llm, language="mr")
    assert draft.language == "en" and "PaisaPath Fintech" in draft.message
