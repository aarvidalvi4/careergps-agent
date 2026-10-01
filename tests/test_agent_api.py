from pathlib import Path

from app.agent.career_agent import TOOLS, run_agent, run_agent_events
from app.llm import BaseLLM, LLMError, LLMResponse, MockLLM, ToolCall

SAMPLES = Path(__file__).parent / "samples"
STRONG = SAMPLES / "strong_data_analyst.pdf"


def tool_calls(result) -> list[str]:
    return [s.tool for s in result.trace if s.type == "tool_call"]


def test_mock_full_flow() -> None:
    result = run_agent(STRONG, "Data Analyst", weeks=8, llm=MockLLM())
    assert result.gap_report and result.roadmap and result.matches and result.outreach
    calls = tool_calls(result)
    assert calls[:2] == ["parse_resume", "lookup_role"]
    assert "draft_outreach" in calls
    assert result.llm_mode == "mock"
    assert result.trace[-1].type == "final"
    assert str(result.gap_report.readiness_score) in result.final_message
    assert not [s for s in result.trace if s.type == "error"]
    assert result.outreach.opening_id == result.matches[0].id


def test_events_stream_steps_then_result() -> None:
    events = list(run_agent_events(STRONG, "Data Analyst", llm=MockLLM()))
    assert all(e["event"] == "step" for e in events[:-1])
    assert events[-1]["event"] == "result"
    gap_step = next(e["data"] for e in events if e["data"].get("tool") == "analyze_gaps" and e["data"]["type"] == "tool_result")
    assert gap_step["summary"].startswith("Readiness ") and "/100" in gap_step["summary"]
    assert [e["data"]["step"] for e in events[:-1]] == list(range(1, len(events)))


def test_tool_schemas_are_complete() -> None:
    for tool in TOOLS.values():
        schema = tool.schema()
        assert schema["description"] and schema["parameters"]["type"] == "object"


class ScriptedLLM(BaseLLM):
    """Plays back a fixed list of replies. is_mock=True so tools use their rule-based paths."""

    is_mock = True

    def __init__(self, replies: list[LLMResponse]) -> None:
        self.replies = replies
        self.seen: list[list[dict]] = []

    def chat(self, messages, system=None, tools=None, max_tokens=2000) -> LLMResponse:
        self.seen.append(list(messages))
        if not self.replies:
            return LLMResponse(text="Done.")
        return self.replies.pop(0)


def call(name: str, n: int, **args) -> LLMResponse:
    return LLMResponse(text=f"Next I'll call {name}.", tool_calls=[ToolCall(id=f"c{n}", name=name, arguments=args)])


def test_recovers_from_tool_errors() -> None:
    llm = ScriptedLLM(
        [
            call("analyze_gaps", 1),
            call("parse_resume", 2),
            call("lookup_role", 3, role="astronaut"),
            call("lookup_role", 4, role="Data Analyst"),
            call("analyze_gaps", 5),
            LLMResponse(text="You're in good shape."),
        ]
    )
    result = run_agent(STRONG, "Data Analyst", llm=llm)
    errors = [s for s in result.trace if s.type == "error"]
    assert len(errors) == 2
    assert "parse_resume first" in errors[0].summary
    assert "Data Analyst" in errors[1].summary  # suggestions / supported roles listed
    assert result.gap_report is not None
    assert result.final_message == "You're in good shape."
    # the error text went back to the model as the tool result
    tool_msgs = [m for m in llm.seen[1] if m["role"] == "tool"]
    assert tool_msgs[0]["content"].startswith("ERROR: No resume parsed yet")


def test_unknown_tool_and_bad_opening_are_errors_not_crashes() -> None:
    llm = ScriptedLLM(
        [
            call("teleport", 1),
            call("parse_resume", 2),
            call("lookup_role", 3, role="data analyst"),
            call("analyze_gaps", 4),
            call("draft_outreach", 5, opening_id="op999"),
        ]
    )
    result = run_agent(STRONG, "Data Analyst", llm=llm)
    errors = [s.summary for s in result.trace if s.type == "error"]
    assert len(errors) == 2
    assert "Unknown tool" in errors[0]
    assert "op01" in errors[1]  # valid ids listed
    assert result.final_message == "Done."


def test_llm_error_falls_back_to_rules() -> None:
    class Broken(BaseLLM):
        is_mock = True

        def chat(self, messages, system=None, tools=None, max_tokens=2000) -> LLMResponse:
            raise LLMError("provider down")

    result = run_agent(STRONG, "Data Analyst", llm=Broken())
    assert result.trace[0].type == "error"
    assert result.trace[-1].type == "final"
    assert result.gap_report is not None and result.outreach is not None


def test_step_limit() -> None:
    llm = ScriptedLLM([call("parse_resume", i) for i in range(20)])
    result = run_agent(STRONG, "Data Analyst", llm=llm)
    assert result.trace[-1].type == "final"
    assert "Stopped after 10 steps" in result.final_message
    assert len(llm.seen) == 10
