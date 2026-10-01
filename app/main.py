"""FastAPI app: API routes and serves frontend/index.html."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent.career_agent import run_agent, run_agent_events
from app.llm import get_llm
from app.models import AgentResult
from app.tools.role_lookup import list_roles

MAX_BYTES = 10 * 1024 * 1024
LANGUAGES = {"en", "hi", "mr"}
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="CareerGPS Agent", description="AI career readiness agent for students in India.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _llm_mode() -> str:
    llm = get_llm()
    return "mock" if llm.is_mock else type(llm).__name__


def _validate(target_role: str, weeks_available: int, language: str) -> str:
    role = target_role.strip()
    if not role:
        raise HTTPException(422, "target_role must not be empty.")
    if not 2 <= weeks_available <= 24:
        raise HTTPException(422, "weeks_available must be between 2 and 24.")
    if language not in LANGUAGES:
        raise HTTPException(422, "language must be one of: en, hi, mr.")
    return role


async def _save_upload(resume: UploadFile) -> str:
    data = await resume.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The resume is larger than 10 MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Upload the resume as a PDF file.")
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _delete(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "llm_mode": _llm_mode()}


@app.get("/roles")
def roles() -> dict[str, list[str]]:
    return {"roles": list_roles()}


@app.post("/analyze", response_model=AgentResult)
async def analyze(
    resume: UploadFile = File(...),
    target_role: str = Form(...),
    weeks_available: int = Form(8),
    language: str = Form("en"),
) -> AgentResult:
    role = _validate(target_role, weeks_available, language)
    path = await _save_upload(resume)
    try:
        return run_agent(path, role, weeks=weeks_available, language=language)  # type: ignore[arg-type]
    finally:
        _delete(path)


@app.post("/analyze/stream")
async def analyze_stream(
    resume: UploadFile = File(...),
    target_role: str = Form(...),
    weeks_available: int = Form(8),
    language: str = Form("en"),
) -> StreamingResponse:
    role = _validate(target_role, weeks_available, language)
    path = await _save_upload(resume)

    def events() -> Iterator[str]:  # sync generator: Starlette runs it in a threadpool
        try:
            for event in run_agent_events(path, role, weeks=weeks_available, language=language):  # type: ignore[arg-type]
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
        except Exception as exc:  # last-resort guard so the client always gets an ending event
            yield f"event: fatal\ndata: {json.dumps({'message': str(exc)})}\n\n"
        finally:
            _delete(path)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
