"""Shared helpers for tools: dataset loading and skill-name normalisation/matching."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.models import RoleRequirements, SkillReq

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@lru_cache(maxsize=1)
def load_roles() -> list[RoleRequirements]:
    with open(DATA_DIR / "roles.json", encoding="utf-8") as f:
        return [RoleRequirements.model_validate(item) for item in json.load(f)]


@lru_cache(maxsize=1)
def load_openings() -> list[dict[str, Any]]:
    path = DATA_DIR / "openings.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_DISALLOWED = re.compile(r"[^a-z0-9+#./]+")


def norm(s: str) -> str:
    """Lowercase, '&' -> ' and ', keep only a-z 0-9 + # . /, collapse whitespace."""
    s = s.lower().replace("&", " and ")
    return " ".join(_DISALLOWED.sub(" ", s).split())


@lru_cache(maxsize=1)
def skill_catalog() -> dict[str, SkillReq]:
    """Canonical skill name -> its definition (names, aliases, related are identical across roles)."""
    catalog: dict[str, SkillReq] = {}
    for role in load_roles():
        for skill in role.required_skills + role.nice_to_have:
            catalog.setdefault(skill.name, skill)
    return catalog


def skill_terms(name: str) -> list[str]:
    """Normalised name + aliases for a canonical skill (just the name if it isn't in the catalog)."""
    skill = skill_catalog().get(name)
    terms = [name, *skill.aliases] if skill else [name]
    return [norm(t) for t in terms]


@lru_cache(maxsize=1)
def alias_map() -> dict[str, str]:
    """Normalised alias (or skill name) -> canonical skill name, across all roles.

    An alias shared by several skills (e.g. "flask" is both REST APIs and Backend
    Framework) maps to the first skill that claims it, in roles.json order.
    """
    mapping: dict[str, str] = {}
    for role in load_roles():
        for skill in role.required_skills + role.nice_to_have:
            for term in [skill.name, *skill.aliases]:
                mapping.setdefault(norm(term), skill.name)
    return mapping


# A term must not be glued to other word characters. A '.' only counts as a
# boundary when it is not part of a token like "node.js" or "asp.net".
_WORD = r"a-z0-9+#"


@lru_cache(maxsize=1)
def _term_patterns() -> list[tuple[str, re.Pattern[str]]]:
    terms = sorted((t for t in alias_map() if len(t) >= 2), key=len, reverse=True)
    return [
        (t, re.compile(rf"(?<![{_WORD}])(?<![{_WORD}]\.){re.escape(t)}(?![{_WORD}])(?!\.[{_WORD}])"))
        for t in terms
    ]


def find_terms_in_text(text: str) -> list[str]:
    """Every known alias that appears as a whole word in free text (normalised form).

    Longest terms are matched first and their span is blanked out, so "node.js"
    does not also report "node". Returned in order of appearance, without duplicates.
    """
    remaining = norm(text)
    found: list[tuple[int, str]] = []
    for term, pattern in _term_patterns():
        for match in pattern.finditer(remaining):
            found.append((match.start(), term))
        remaining = pattern.sub(lambda m: " " * len(m.group()), remaining)
    seen: set[str] = set()
    ordered: list[str] = []
    for _, term in sorted(found):
        if term not in seen:
            seen.add(term)
            ordered.append(term)
    return ordered
