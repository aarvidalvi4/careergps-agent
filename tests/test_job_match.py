from pathlib import Path

import pytest

from app.llm import MockLLM
from app.models import GapReport, ResumeProfile, RoleRequirements
from app.tools.gap_analysis import analyze_gaps
from app.tools.job_match import OpeningNotFound, get_opening, match_jobs
from app.tools.resume_parser import parse_resume
from app.tools.role_lookup import lookup_role
from app.tools.skills import load_openings, skill_catalog

SAMPLES = Path(__file__).parent / "samples"


def analysed(sample: str, role: str) -> tuple[ResumeProfile, GapReport, RoleRequirements]:
    profile = parse_resume(SAMPLES / f"{sample}.pdf", MockLLM())
    req = lookup_role(role)
    return profile, analyze_gaps(profile, req, MockLLM()), req


def test_openings_use_canonical_skill_names() -> None:
    canonical = set(skill_catalog())
    openings = load_openings()
    assert len(openings) == 20
    for opening in openings:
        assert 3 <= len(opening.skills) <= 5, opening.id
        assert set(opening.skills) <= canonical, (opening.id, set(opening.skills) - canonical)
        assert opening.link == f"https://example.com/apply/{opening.id}"
    assert {o.family for o in openings} == {"data", "ml", "web", "security", "business"}


def test_strong_data_analyst_top_match() -> None:
    top = match_jobs(*analysed("strong_data_analyst", "Data Analyst"))[0]
    assert top.match_percent >= 75
    assert get_opening(top.id).family == "data"
    assert "analyst" in top.role.lower()


def test_matches_are_ranked_and_limited() -> None:
    matches = match_jobs(*analysed("medium_data_analyst", "Data Analyst"), limit=3)
    assert len(matches) == 3
    assert matches == match_jobs(*analysed("medium_data_analyst", "Data Analyst"), limit=3)  # deterministic


def test_off_role_skills_checked_on_resume() -> None:
    # HTML isn't an ML Engineer skill, but the weak resume lists it: credit it, don't call it missing.
    matches = match_jobs(*analysed("weak_ml_engineer", "ML Engineer"), limit=20)
    frontend = next(m for m in matches if m.id == "op12")
    assert "HTML" in frontend.matched_skills and frontend.match_percent == 25


def test_partial_counts_half() -> None:
    profile, gaps, req = analysed("medium_data_analyst", "Data Analyst")
    mis = next(m for m in match_jobs(profile, gaps, req, limit=20) if m.id == "op03")
    # Excel matched (1) + Data Visualization partial (0.5) + Data Cleaning partial (0.5) = 2 / 3
    assert mis.match_percent == 67
    assert mis.matched_skills == ["Excel"]


def test_get_opening() -> None:
    assert get_opening("op01").company == "Kisanlytics"
    with pytest.raises(OpeningNotFound):
        get_opening("op99")
