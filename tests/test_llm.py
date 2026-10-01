import json
from typing import Any

import httpx
import pytest

from app import llm
from app.llm import (
    AnthropicLLM,
    BaseLLM,
    LLMError,
    LLMResponse,
    MockLLM,
    OpenAICompatLLM,
    ToolCall,
    extract_json,
    get_llm,
)

TOOLS = [{"name": n, "description": "", "parameters": {"type": "object"}} for n in MockLLM.TOOL_ORDER]


class FakePost:
    """Stands in for httpx.post: records the request and returns a canned response."""

    def __init__(self, status: int = 200, body: Any = None, text: str | None = None) -> None:
        self.status, self.body, self.text = status, body, text
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, headers: dict, json: dict, timeout: float) -> httpx.Response:
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        request = httpx.Request("POST", url)
        if self.text is not None:
            return httpx.Response(self.status, text=self.text, request=request)
        return httpx.Response(self.status, json=self.body, request=request)


# --- extract_json ---------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ("```\n[1, 2]\n```", [1, 2]),
        ('Here you go: {"a": [1]} hope that helps', {"a": [1]}),
        ("Sure: [1, 2] done", [1, 2]),
        ('Matches: [{"id": 1}, {"id": 2}]', [{"id": 1}, {"id": 2}]),
    ],
)
def test_extract_json(text: str, expected: Any) -> None:
    assert extract_json(text) == expected


def test_extract_json_raises_on_garbage() -> None:
    with pytest.raises(LLMError):
        extract_json("no json here")


# --- complete_json --------------------------------------------------------


class ScriptedLLM(BaseLLM):
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.seen: list[list[dict]] = []

    def chat(self, messages, system=None, tools=None, max_tokens=2000) -> LLMResponse:
        self.seen.append(list(messages))
        return LLMResponse(text=self.replies.pop(0))


def test_complete_json_first_try() -> None:
    fake = ScriptedLLM(['{"ok": true}'])
    assert fake.complete_json("q") == {"ok": True}
    assert len(fake.seen) == 1


def test_complete_json_retries_once_then_succeeds() -> None:
    fake = ScriptedLLM(["sorry, no", '{"ok": true}'])
    assert fake.complete_json("q") == {"ok": True}
    assert len(fake.seen) == 2
    assert "not valid JSON" in fake.seen[1][-1]["content"]


def test_complete_json_raises_after_retry() -> None:
    fake = ScriptedLLM(["nope", "still nope"])
    with pytest.raises(LLMError):
        fake.complete_json("q")


# --- MockLLM --------------------------------------------------------------


def test_mock_without_tools_returns_empty_json() -> None:
    m = MockLLM()
    assert m.is_mock
    assert m.chat([{"role": "user", "content": "hi"}]).text == "{}"
    assert m.complete_json("anything") == {}


def test_mock_walks_tools_in_order() -> None:
    results = {
        "parse_resume": "{}",
        "lookup_role": "{}",
        "analyze_gaps": json.dumps({"score": 42, "band": "Developing", "missing_skills": [{"skill": "SQL"}]}),
        "build_roadmap": "{}",
        "match_jobs": json.dumps({"jobs": [{"id": "job-7"}, {"id": "job-9"}]}),
        "draft_outreach": "{}",
    }
    messages: list[dict] = [{"role": "user", "content": "Target role: Data Analyst\nWeeks available: 6"}]
    called: list[tuple[str, dict]] = []
    m = MockLLM()
    for _ in range(10):
        reply = m.chat(messages, tools=TOOLS)
        if not reply.tool_calls:
            break
        assert reply.text  # every step has a thought
        messages.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})
        for call in reply.tool_calls:
            called.append((call.name, call.arguments))
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": results[call.name]})

    assert [name for name, _ in called] == MockLLM.TOOL_ORDER
    args = dict(called)
    assert args["lookup_role"] == {"role": "Data Analyst"}
    assert args["build_roadmap"] == {"weeks": 6}
    assert args["draft_outreach"] == {"job_id": "job-7"}
    assert "42" in reply.text and "Developing" in reply.text and "SQL" in reply.text


def test_mock_skips_tools_not_offered() -> None:
    tools = [t for t in TOOLS if t["name"] != "parse_resume"]
    reply = MockLLM().chat([{"role": "user", "content": "Target role: SDE"}], tools=tools)
    assert reply.tool_calls[0].name == "lookup_role"


# --- AnthropicLLM ---------------------------------------------------------


def test_anthropic_groups_consecutive_tool_results() -> None:
    calls = [ToolCall("a", "x", {}), ToolCall("b", "y", {"k": 1})]
    converted = AnthropicLLM._convert_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "thinking", "tool_calls": calls},
            {"role": "tool", "tool_call_id": "a", "name": "x", "content": "1"},
            {"role": "tool", "tool_call_id": "b", "name": "y", "content": "2"},
        ]
    )
    assert [m["role"] for m in converted] == ["user", "assistant", "user"]
    assert converted[1]["content"] == [
        {"type": "text", "text": "thinking"},
        {"type": "tool_use", "id": "a", "name": "x", "input": {}},
        {"type": "tool_use", "id": "b", "name": "y", "input": {"k": 1}},
    ]
    assert converted[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "a", "content": "1"},
        {"type": "tool_result", "tool_use_id": "b", "content": "2"},
    ]


def test_anthropic_chat_request_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePost(
        body={
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "tu_1", "name": "lookup_role", "input": {"role": "SDE"}},
            ]
        }
    )
    monkeypatch.setattr(llm.httpx, "post", fake)
    tools = [{"name": "lookup_role", "description": "d", "parameters": {"type": "object"}}]
    reply = AnthropicLLM("key", "m").chat([{"role": "user", "content": "hi"}], system="sys", tools=tools)

    req = fake.calls[0]
    assert req["url"] == "https://api.anthropic.com/v1/messages"
    assert req["headers"]["x-api-key"] == "key"
    assert req["headers"]["anthropic-version"] == "2023-06-01"
    assert req["timeout"] == 90
    assert req["json"]["system"] == "sys"
    assert req["json"]["tools"] == [{"name": "lookup_role", "description": "d", "input_schema": {"type": "object"}}]
    assert reply.text == "Let me check."
    assert reply.tool_calls == [ToolCall("tu_1", "lookup_role", {"role": "SDE"})]


def test_anthropic_non_200_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.httpx, "post", FakePost(status=401, text="x" * 1000))
    with pytest.raises(LLMError) as exc:
        AnthropicLLM("key").chat([{"role": "user", "content": "hi"}])
    assert "401" in str(exc.value)
    assert "x" * 300 in str(exc.value) and "x" * 301 not in str(exc.value)


# --- OpenAICompatLLM ------------------------------------------------------


def test_openai_message_conversion() -> None:
    calls = [ToolCall("c1", "x", {"k": 1})]
    converted = OpenAICompatLLM._convert_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "tool_call_id": "c1", "name": "x", "content": "ok"},
        ],
        system="sys",
    )
    assert converted[0] == {"role": "system", "content": "sys"}
    assert converted[2]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "x", "arguments": '{"k": 1}'}}
    ]
    assert converted[3] == {"role": "tool", "tool_call_id": "c1", "content": "ok"}


def test_openai_chat_parses_bad_arguments_as_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePost(
        body={
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "c1", "type": "function", "function": {"name": "a", "arguments": '{"x": 1}'}},
                            {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{broken"}},
                        ],
                    }
                }
            ]
        }
    )
    monkeypatch.setattr(llm.httpx, "post", fake)
    reply = OpenAICompatLLM("key", "m", "https://example.test/v1/").chat([{"role": "user", "content": "hi"}])

    assert fake.calls[0]["url"] == "https://example.test/v1/chat/completions"
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer key"
    assert reply.text == ""
    assert reply.tool_calls == [ToolCall("c1", "a", {"x": 1}), ToolCall("c2", "b", {})]


def test_openai_network_error_raises_llmerror(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(llm.httpx, "post", boom)
    with pytest.raises(LLMError):
        OpenAICompatLLM("key").chat([{"role": "user", "content": "hi"}])


# --- get_llm --------------------------------------------------------------


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)  # ignore the developer's .env
    for var in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_get_llm_mock_when_no_key(env: pytest.MonkeyPatch) -> None:
    env.setenv("LLM_PROVIDER", "anthropic")
    assert isinstance(get_llm(), MockLLM)


def test_get_llm_anthropic_default_model(env: pytest.MonkeyPatch) -> None:
    env.setenv("LLM_PROVIDER", "anthropic")
    env.setenv("LLM_API_KEY", "k")
    client = get_llm()
    assert isinstance(client, AnthropicLLM) and client.model == "claude-sonnet-5-5"


def test_get_llm_openai_with_base_url(env: pytest.MonkeyPatch) -> None:
    env.setenv("LLM_PROVIDER", "openai")
    env.setenv("LLM_API_KEY", "k")
    env.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    client = get_llm()
    assert isinstance(client, OpenAICompatLLM)
    assert client.model == "gpt-4o-mini" and client.base_url == "https://api.groq.com/openai/v1"


def test_get_llm_unknown_provider(env: pytest.MonkeyPatch) -> None:
    env.setenv("LLM_PROVIDER", "bogus")
    env.setenv("LLM_API_KEY", "k")
    with pytest.raises(LLMError):
        get_llm()
