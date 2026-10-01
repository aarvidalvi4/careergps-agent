# CareerGPS Agent

An AI career readiness agent for students at tier-2/tier-3 colleges in India. A student uploads a resume (PDF) and states a target role; the agent analyses the resume, scores readiness against the role, finds skill gaps, and produces a concrete plan to close them.

Python 3.12, FastAPI, Pydantic v2. The frontend is a single static HTML page served by FastAPI.

## Layout

```
app/main.py         FastAPI app (API routes + serves frontend/index.html)
app/llm.py          the ONLY module that talks to an LLM provider
app/models.py       Pydantic models for all tool inputs/outputs
app/agent/          agent loop (decides which tools to call, in what order)
app/tools/          one file per tool
data/               JSON datasets (roles, skills, resources, ...)
frontend/           single index.html, served by FastAPI
tests/              pytest tests
tests/samples/      sample resume PDFs
```

## Conventions

- **Type hints everywhere.** Every tool takes and returns Pydantic models defined in `app/models.py`; no loose dicts crossing tool boundaries.
- **Scores are deterministic.** All scores are computed in Python from the data. The LLM only judges ambiguous skill matches (e.g. is "ReactJS" the same as "React.js"?) and writes natural language (summaries, explanations, advice). It never produces a number that ends up in a score.
- **Every tool has a rule-based fallback.** If the LLM call fails, times out, or returns unparseable output, the tool falls back to a rule-based path (exact/normalised matching, templated text) so the app degrades instead of crashing.
- **All LLM access goes through `app/llm.py`.** No other module imports a provider SDK or makes provider HTTP calls. Provider is chosen by `LLM_PROVIDER` (`anthropic` | `openai` | `mock`); `openai` means any OpenAI-compatible API (Groq, Gemini, OpenRouter) via `LLM_BASE_URL`.
- **Tests run with `LLM_PROVIDER=mock` and need no API key.** No test may hit the network.

## Commands

```
uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt   # install
LLM_PROVIDER=mock .venv/Scripts/python.exe -m pytest                       # tests
.venv/Scripts/python.exe -m uvicorn app.main:app --reload                  # dev server
```

Copy `.env.example` to `.env` to configure the LLM provider.
