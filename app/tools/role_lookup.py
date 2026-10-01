"""Resolve what the student typed as a target role to a role in data/roles.json."""

from __future__ import annotations

import difflib
import re

from app.models import RoleRequirements
from app.tools.skills import load_roles, norm


class RoleNotFound(Exception):
    def __init__(self, query: str, suggestions: list[str]) -> None:
        self.query = query
        self.suggestions = suggestions
        super().__init__(f"No role matches {query!r}. Did you mean: {', '.join(suggestions)}?")


def list_roles() -> list[str]:
    return [role.role for role in load_roles()]


def _names() -> dict[str, RoleRequirements]:
    """Normalised role name or alias -> role."""
    names: dict[str, RoleRequirements] = {}
    for role in load_roles():
        for name in [role.role, *role.aliases]:
            names.setdefault(norm(name), role)
    return names


def _contains_words(haystack: str, needle: str) -> bool:
    return re.search(rf"(?<!\S){re.escape(needle)}(?!\S)", haystack) is not None


def _suggestions(query: str, n: int = 3) -> list[str]:
    best: dict[str, float] = {}
    for name, role in _names().items():
        score = difflib.SequenceMatcher(None, query, name).ratio()
        best[role.role] = max(best.get(role.role, 0.0), score)
    return sorted(best, key=lambda r: -best[r])[:n]


def lookup_role(query: str) -> RoleRequirements:
    """Match by exact name/alias, then a close spelling, then substring; else raise RoleNotFound."""
    q = norm(query)
    names = _names()

    role = names.get(q)

    if role is None and q:
        close = difflib.get_close_matches(q, list(names), n=1, cutoff=0.75)
        if close:
            role = names[close[0]]

    if role is None and q:
        # A known name inside the query: "data analyst intern" -> Data Analyst (longest name wins).
        inside = sorted((n for n in names if _contains_words(q, n)), key=len, reverse=True)
        if inside:
            role = names[inside[0]]
        else:
            # The query inside a known name, only if it points to a single role: "frontend" -> Frontend Developer.
            hits = {names[n].role: names[n] for n in names if _contains_words(n, q)}
            if len(hits) == 1:
                role = next(iter(hits.values()))

    if role is None:
        raise RoleNotFound(query, _suggestions(q))
    return role.model_copy(deep=True)
