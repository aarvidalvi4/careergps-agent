import pytest

from app.tools.role_lookup import RoleNotFound, list_roles, lookup_role
from app.tools.skills import alias_map, find_terms_in_text, load_roles, norm

# --- skills helpers -------------------------------------------------------


def test_roles_load_with_aliases() -> None:
    roles = load_roles()
    assert len(roles) == 8
    assert "mis analyst" in next(r for r in roles if r.role == "Data Analyst").aliases


def test_norm() -> None:
    assert norm("  Data Structures & Algorithms ") == "data structures and algorithms"
    assert norm("C++, C#, Node.js; TCP/IP!") == "c++ c# node.js tcp/ip"


def test_alias_map_includes_names_and_aliases() -> None:
    m = alias_map()
    assert m["sql"] == "SQL"
    assert m["power bi"] == "Data Visualization"
    assert m["reactjs"] == "React"


def test_find_terms_whole_words_longest_first() -> None:
    found = find_terms_in_text("Built dashboards in Power BI with Node.js, HTML/CSS and C. Typescript too.")
    assert "power bi" in found and "node.js" in found and "html" in found and "css" in found
    assert "node" not in found  # consumed by node.js
    assert "ts" not in found  # not a whole word inside "typescript"
    assert "c" not in found


def test_find_terms_ignores_partial_words() -> None:
    assert "sql" not in find_terms_in_text("I like nosqlish things")
    assert find_terms_in_text("Learning Python.") == ["python"]


# --- role lookup ----------------------------------------------------------


def test_list_roles() -> None:
    assert "Data Analyst" in list_roles() and len(list_roles()) == 8


@pytest.mark.parametrize(
    "query, expected",
    [
        ("data analyst intern", "Data Analyst"),
        ("machine learning engineer", "ML Engineer"),
        ("Data Analist", "Data Analyst"),
        ("Data Analyst", "Data Analyst"),
        ("MIS Analyst", "Data Analyst"),
        ("frontend", "Frontend Developer"),
    ],
)
def test_lookup_role(query: str, expected: str) -> None:
    assert lookup_role(query).role == expected


def test_lookup_role_not_found_has_suggestions() -> None:
    with pytest.raises(RoleNotFound) as exc:
        lookup_role("astronaut")
    assert len(exc.value.suggestions) == 3
    assert all(s in list_roles() for s in exc.value.suggestions)


def test_lookup_role_returns_a_copy() -> None:
    lookup_role("Data Analyst").required_skills.clear()
    assert lookup_role("Data Analyst").required_skills
