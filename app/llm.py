"""The ONLY module that talks to an LLM provider (anthropic | openai-compatible | mock).

Everything outside this module uses a neutral message format:

    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

and tools are described as {"name": str, "description": str, "parameters": <JSON schema>}.
Each provider class converts to and from its own wire format.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

TIMEOUT_SECONDS = 90.0
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

Message = dict[str, Any]
Tool = dict[str, Any]


class LLMError(Exception):
    """Any failure talking to the LLM or interpreting its reply."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply.

    Tries, in order: the whole text, the contents of a ```json fence, and the
    first {...} or [...] span. Raises LLMError if nothing parses.
    """
    if text is None:
        raise LLMError("No text to parse as JSON")
    candidates = [text.strip()]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    raise LLMError(f"Could not parse JSON from model reply: {text[:300]!r}")


def _raise_for_status(provider: str, response: httpx.Response) -> None:
    if response.status_code != 200:
        raise LLMError(f"{provider} API error {response.status_code}: {response.text[:300]}")


class BaseLLM:
    is_mock: bool = False

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Tool] | None = None,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        raise NotImplementedError

    def complete_json(self, prompt: str, system: str | None = None) -> Any:
        """Ask for a JSON-only reply and parse it; retry once with a stricter nudge."""
        json_system = ((system + "\n\n") if system else "") + (
            "Respond with valid JSON only. No prose, no explanations, no markdown fences."
        )
        messages: list[Message] = [{"role": "user", "content": prompt}]
        reply = self.chat(messages, system=json_system)
        try:
            return extract_json(reply.text)
        except LLMError:
            pass
        messages += [
            {"role": "assistant", "content": reply.text, "tool_calls": []},
            {
                "role": "user",
                "content": "That was not valid JSON. Reply again with ONLY the JSON value, "
                "starting with { or [ and nothing before or after it.",
            },
        ]
        retry = self.chat(messages, system=json_system)
        return extract_json(retry.text)  # raises LLMError on second failure

    def _post(self, provider: str, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = httpx.post(url, headers=headers, json=body, timeout=TIMEOUT_SECONDS)
        except httpx.HTTPError as exc:
            raise LLMError(f"{provider} request failed: {exc}") from exc
        _raise_for_status(provider, response)
        try:
            return response.json()
        except ValueError as exc:
            raise LLMError(f"{provider} returned non-JSON body: {response.text[:300]}") from exc


class AnthropicLLM(BaseLLM):
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL) -> None:
        self.api_key = api_key
        self.model = model

    @staticmethod
    def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush_results() -> None:
            # Anthropic requires all tool results answering one assistant turn in ONE user message.
            if pending_results:
                out.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for msg in messages:
            role = msg["role"]
            if role == "tool":
                pending_results.append(
                    {"type": "tool_result", "tool_use_id": msg["tool_call_id"], "content": msg.get("content", "")}
                )
                continue
            flush_results()
            if role == "assistant":
                blocks: list[dict[str, Any]] = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for call in msg.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
                if not blocks:
                    blocks.append({"type": "text", "text": "(no content)"})
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": "user", "content": msg.get("content", "")})
        flush_results()
        return out

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Tool] | None = None,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": self._convert_messages(messages),
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [
                {"name": t["name"], "description": t.get("description", ""), "input_schema": t["parameters"]}
                for t in tools
            ]
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = self._post("Anthropic", self.URL, headers, body)

        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                args = block.get("input")
                calls.append(ToolCall(id=block["id"], name=block["name"], arguments=args if isinstance(args, dict) else {}))
        return LLMResponse(text="".join(texts), tool_calls=calls)


class OpenAICompatLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_MODEL, base_url: str = DEFAULT_OPENAI_BASE_URL) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def _convert_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for msg in messages:
            role = msg["role"]
            if role == "tool":
                out.append({"role": "tool", "tool_call_id": msg["tool_call_id"], "content": msg.get("content", "")})
            elif role == "assistant":
                converted: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or None}
                calls = msg.get("tool_calls") or []
                if calls:
                    converted["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in calls
                    ]
                elif converted["content"] is None:
                    converted["content"] = ""
                out.append(converted)
            else:
                out.append({"role": "user", "content": msg.get("content", "")})
        return out

    @staticmethod
    def _parse_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw or "{}")
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Tool] | None = None,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": self._convert_messages(messages, system),
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t["parameters"],
                    },
                }
                for t in tools
            ]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = self._post("OpenAI-compatible", f"{self.base_url}/chat/completions", headers, body)

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenAI-compatible response: {str(data)[:300]}") from exc
        calls = [
            ToolCall(
                id=c.get("id") or f"call_{i}",
                name=c.get("function", {}).get("name", ""),
                arguments=self._parse_arguments(c.get("function", {}).get("arguments")),
            )
            for i, c in enumerate(message.get("tool_calls") or [])
        ]
        return LLMResponse(text=message.get("content") or "", tool_calls=calls)


class MockLLM(BaseLLM):
    """Deterministic, offline stand-in that walks the agent's tools in a fixed order."""

    is_mock = True

    TOOL_ORDER = ["parse_resume", "lookup_role", "analyze_gaps", "build_roadmap", "match_jobs", "draft_outreach"]
    THOUGHTS = {
        "parse_resume": "First I'll read the resume to see what the student already has.",
        "lookup_role": "Next I'll look up what the target role actually requires.",
        "analyze_gaps": "Now I'll compare the resume against the role to find the gaps.",
        "build_roadmap": "With the gaps known, I'll build a week-by-week plan to close them.",
        "match_jobs": "Let me find openings that fit the student's current profile.",
        "draft_outreach": "Finally I'll draft an outreach message for the best-matching job.",
    }

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Tool] | None = None,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        if not tools:
            return LLMResponse(text="{}")

        offered = {t["name"] for t in tools}
        results = {m.get("name"): m.get("content", "") for m in messages if m.get("role") == "tool"}
        first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        role_match = re.search(r"Target role:\s*(.+)", first_user)
        weeks_match = re.search(r"Weeks available:\s*(\d+)", first_user)
        target_role = role_match.group(1).strip() if role_match else ""
        weeks = int(weeks_match.group(1)) if weeks_match else 8

        for step, name in enumerate(self.TOOL_ORDER):
            if name not in offered or name in results:
                continue
            args: dict[str, Any] = {}
            if name in ("lookup_role", "match_jobs"):
                args = {"role": target_role}
            elif name == "build_roadmap":
                args = {"weeks": weeks}
            elif name == "draft_outreach":
                opening_id = _first_id(_safe_json(results.get("match_jobs", "")))
                if opening_id is not None:
                    args = {"opening_id": opening_id}
            return LLMResponse(
                text=self.THOUGHTS[name],
                tool_calls=[ToolCall(id=f"mock_{step}_{name}", name=name, arguments=args)],
            )

        return LLMResponse(text=self._final_message(_safe_json(results.get("analyze_gaps", ""))))

    @staticmethod
    def _final_message(gaps: Any) -> str:
        if not isinstance(gaps, dict):
            return "I've finished the analysis. See the roadmap and job matches below."
        score = gaps.get("score", gaps.get("readiness_score"))
        band = gaps.get("band", "")
        missing = gaps.get("missing_skills") or gaps.get("gaps") or []
        top = missing[0] if missing else None
        if isinstance(top, dict):
            top = top.get("skill") or top.get("name")
        if score is None:
            parts = ["I've finished the analysis."]
        else:
            parts = [f"Your readiness score is {score}" + (f" ({band})" if band else "") + "."]
        if top:
            parts.append(f"The most important skill to work on next is {top}.")
        parts.append("Your roadmap and matching jobs are below.")
        return " ".join(parts)


def _safe_json(text: str) -> Any:
    try:
        return extract_json(text)
    except LLMError:
        return None


def _first_id(value: Any) -> Any:
    """Depth-first search for the first "id" field in a parsed JSON value."""
    if isinstance(value, dict):
        if "id" in value:
            return value["id"]
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return None
    for child in children:
        found = _first_id(child)
        if found is not None:
            return found
    return None


def get_llm() -> BaseLLM:
    """Build the configured LLM client. Falls back to MockLLM when provider is mock or no key is set."""
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip()

    if provider == "mock" or not api_key:
        return MockLLM()
    if provider == "anthropic":
        return AnthropicLLM(api_key, model or DEFAULT_ANTHROPIC_MODEL)
    if provider == "openai":
        return OpenAICompatLLM(api_key, model or DEFAULT_OPENAI_MODEL, base_url or DEFAULT_OPENAI_BASE_URL)
    raise LLMError(f"Unknown LLM_PROVIDER {provider!r}; expected anthropic, openai or mock")
