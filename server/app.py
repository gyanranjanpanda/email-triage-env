"""
FastAPI server for the Email Triage Environment.

Exposes the OpenEnv-compliant REST API:
- POST /reset       — Start a new episode
- POST /step        — Execute an action
- GET  /state       — Get current episode state
- GET  /health      — Health check
- GET  /tasks       — List available tasks
- POST /grader      — Get grader score after episode
- POST /baseline    — Run baseline inference and return scores
"""

from __future__ import annotations

import sys
import os

# Ensure parent directory is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models import TriageAction, TriageObservation, TriageState
from server.environment import EmailTriageEnvironment, TASKS


# ─── Request/Response Models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = Field(default="easy_triage", description="Task to run")
    seed: Optional[int] = Field(default=None, description="Random seed")


class StepRequest(BaseModel):
    email_id: str = Field(..., description="Email ID to process")
    category: str = Field(..., description="Classified category")
    priority: str = Field(..., description="Assigned priority")
    response_draft: str = Field(default="", description="Draft response")
    response_tone: str = Field(default="neutral", description="Response tone")
    should_respond: bool = Field(default=True, description="Whether to respond")
    escalate: bool = Field(default=False, description="Whether to escalate")
    tags: List[str] = Field(default_factory=list, description="Optional tags")


class HealthResponse(BaseModel):
    status: str = "healthy"
    environment: str = "email_triage"
    version: str = "1.0.0"


class TaskInfo(BaseModel):
    id: str
    name: str
    description: str
    difficulty: str
    num_emails: int
    max_steps: int
    action_schema: Dict[str, Any]


# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Email Triage Environment",
    description="An OpenEnv-compliant environment for email triage and response training",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = EmailTriageEnvironment()


# ─── Action Schema ────────────────────────────────────────────────────────────

ACTION_SCHEMA = {
    "type": "object",
    "required": ["email_id", "category", "priority"],
    "properties": {
        "email_id": {
            "type": "string",
            "description": "ID of the email being acted upon",
        },
        "category": {
            "type": "string",
            "enum": ["urgent_bug", "feature_request", "billing", "general_inquiry", "spam", "complaint", "partnership", "internal"],
            "description": "Classified category",
        },
        "priority": {
            "type": "string",
            "enum": ["critical", "high", "medium", "low"],
            "description": "Assigned priority",
        },
        "response_draft": {
            "type": "string",
            "description": "Draft response text (empty if no response needed)",
            "default": "",
        },
        "response_tone": {
            "type": "string",
            "enum": ["formal", "friendly", "apologetic", "neutral"],
            "description": "Tone of the response",
            "default": "neutral",
        },
        "should_respond": {
            "type": "boolean",
            "description": "Whether this email requires a response",
            "default": True,
        },
        "escalate": {
            "type": "boolean",
            "description": "Whether to escalate to a human supervisor",
            "default": False,
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional tags for organization",
            "default": [],
        },
    },
}


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint — environment info and available endpoints."""
    return {
        "name": "Email Triage & Response Environment",
        "version": "1.0.0",
        "description": (
            "An OpenEnv-compliant RL environment for training AI agents on "
            "email triage tasks. Agents classify, prioritize, and respond to emails."
        ),
        "endpoints": {
            "GET /": "This page",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /tasks": "List available tasks with action schema",
            "GET /state": "Current episode state",
            "POST /reset": "Start a new episode (body: {task_id, seed})",
            "POST /step": "Execute a triage action on one email",
            "POST /grader": "Get grader score after episode completes",
            "POST /baseline": "Run baseline agent on all tasks",
        },
        "tasks": list(TASKS.keys()),
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse()


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment and start a new episode.

    Returns the initial observation with emails to process.
    """
    try:
        obs = env.reset(task_id=request.task_id, seed=request.seed)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    """
    Execute a triage action on one email.

    Returns observation with feedback, reward, and remaining emails.
    """
    try:
        action = TriageAction(
            email_id=request.email_id,
            category=request.category,
            priority=request.priority,
            response_draft=request.response_draft,
            response_tone=request.response_tone,
            should_respond=request.should_respond,
            escalate=request.escalate,
            tags=request.tags,
        )
        obs = env.step(action)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    """Get current episode state and metadata."""
    try:
        return env.state.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def tasks():
    """List available tasks with their action schemas."""
    task_list = []
    for task_id, task in TASKS.items():
        task_list.append(TaskInfo(
            id=task_id,
            name=task["name"],
            description=task["description"],
            difficulty=task["difficulty"],
            num_emails=task["num_emails"],
            max_steps=task["max_steps"],
            action_schema=ACTION_SCHEMA,
        ).model_dump())
    return {"tasks": task_list}


@app.post("/grader")
async def grader():
    """
    Run the grader on the completed episode.

    Returns detailed scoring breakdown.
    """
    try:
        if not env.state.done and env.state.step_count == 0:
            raise HTTPException(
                status_code=400,
                detail="No episode has been run yet. Call /reset then /step first.",
            )
        result = env.get_grader_result()
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/baseline")
async def baseline():
    """
    Run the baseline inference script and return scores for all tasks.

    Uses a simple heuristic agent as baseline.
    """
    try:
        from baseline_agent import run_baseline
        results = run_baseline(env)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Main entry point ────────────────────────────────────────────────────────


def main():
    """Entry point for the server, callable via `python -m server.app` or project scripts."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
