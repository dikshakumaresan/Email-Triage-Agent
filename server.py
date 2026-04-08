"""
server.py
---------
FastAPI web server that exposes the Email Triage Environment
via HTTP endpoints following the OpenEnv spec.

Endpoints:
  POST /reset          -> Start a new episode, get first observation
  POST /step           -> Submit an action, get next observation + reward
  GET  /state          -> Get current environment state
  GET  /tasks          -> List all available tasks
  GET  /health         -> Health check (used by HF Space validator)

Usage:
  uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import EmailTriageEnv, StepResult, ResetResult, EnvironmentState


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="Email Triage Environment",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to triage emails. "
        "Supports 3 tasks: priority classification (easy), "
        "routing + reply (medium), and multi-thread analysis (hard)."
    ),
    version="1.0.0",
)

# Allow all origins — needed for HuggingFace Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL ENVIRONMENT INSTANCES
# One env instance per task — shared across requests
# ============================================================

# We keep one active env per task name
_envs: dict[str, EmailTriageEnv] = {}

DEFAULT_TASK = "task1_priority_classification"

VALID_TASK_NAMES = [
    "task1_priority_classification",
    "task2_routing_and_reply",
    "task3_thread_analysis",
]


def get_env(task_name: str) -> EmailTriageEnv:
    """Get or create an environment instance for the given task."""
    if task_name not in VALID_TASK_NAMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown task '{task_name}'. "
                f"Valid tasks: {VALID_TASK_NAMES}"
            ),
        )
    if task_name not in _envs:
        _envs[task_name] = EmailTriageEnv(task_name=task_name)
    return _envs[task_name]


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ResetRequest(BaseModel):
    """Request body for /reset"""
    task_name: Optional[str] = DEFAULT_TASK


class StepRequest(BaseModel):
    """
    Request body for /step
    The action dict fields depend on the task:

    Task 1: priority, category
    Task 2: priority, category, route_to, draft_reply
    Task 3: priority, category, route_to, required_actions,
            draft_reply, context_summary
    """
    task_name: Optional[str] = DEFAULT_TASK
    action: dict


class ResetResponse(BaseModel):
    """Response from /reset"""
    observation: dict
    info: dict


class StepResponse(BaseModel):
    """Response from /step"""
    observation: Optional[dict]
    reward: float
    done: bool
    info: dict


class StateResponse(BaseModel):
    """Response from /state"""
    task_name: str
    current_step: int
    max_steps: int
    total_reward: float
    episode_rewards: list
    done: bool
    emails_processed: int
    emails_remaining: int


class HealthResponse(BaseModel):
    """Response from /health"""
    status: str
    environment: str
    version: str
    available_tasks: list


class TasksResponse(BaseModel):
    """Response from /tasks"""
    tasks: list


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Used by HuggingFace Space validator to confirm the server is live.
    """
    return HealthResponse(
        status="ok",
        environment="email-triage-env",
        version="1.0.0",
        available_tasks=VALID_TASK_NAMES,
    )


@app.get("/tasks", response_model=TasksResponse)
def list_tasks():
    """
    List all available tasks with their descriptions and difficulty levels.
    """
    env = get_env(DEFAULT_TASK)
    return TasksResponse(tasks=env.list_available_tasks())


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = None):
    """
    Reset the environment and start a new episode.

    Returns the first email/thread as the initial observation,
    along with task instructions.

    Body (optional):
        task_name: which task to run (default: task1_priority_classification)
    """
    # Handle empty body (validator sends empty POST to /reset)
    if request is None:
        request = ResetRequest()

    task_name = request.task_name or DEFAULT_TASK
    env = get_env(task_name)

    result: ResetResult = env.reset()

    return ResetResponse(
        observation=result.observation,
        info=result.info,
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Submit one action and get the next observation + reward.

    The agent reads the current email/thread from the last observation,
    decides what to do, and submits an action here.

    Body:
        task_name: which task is running
        action:    dict with the agent's answer (fields depend on task)

    Example action for Task 1:
        {"priority": "critical", "category": "technical_incident"}

    Example action for Task 2:
        {
          "priority": "high",
          "category": "billing",
          "route_to": "finance_team",
          "draft_reply": "Thank you for reaching out..."
        }

    Example action for Task 3:
        {
          "priority": "high",
          "category": "account_management",
          "route_to": "account_management_team",
          "required_actions": ["reverse_cancellation", "process_upgrade"],
          "draft_reply": "We have noted your request...",
          "context_summary": "Client initially cancelled but changed their mind..."
        }
    """
    task_name = request.task_name or DEFAULT_TASK
    env = get_env(task_name)

    result: StepResult = env.step(request.action)

    return StepResponse(
        observation=result.observation,
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/state", response_model=StateResponse)
def state(task_name: str = DEFAULT_TASK):
    """
    Get the current internal state of the environment.

    Query param:
        task_name: which task to inspect (default: task1_priority_classification)
    """
    env = get_env(task_name)
    s: EnvironmentState = env.state()

    return StateResponse(
        task_name=s.task_name,
        current_step=s.current_step,
        max_steps=s.max_steps,
        total_reward=s.total_reward,
        episode_rewards=s.episode_rewards,
        done=s.done,
        emails_processed=s.emails_processed,
        emails_remaining=s.emails_remaining,
    )


@app.get("/")
def root():
    """Root endpoint — shows environment info."""
    return {
        "name": "Email Triage Environment",
        "description": (
            "An OpenEnv-compliant environment where AI agents learn to triage emails. "
            "3 tasks ranging from easy priority classification to hard multi-thread analysis."
        ),
        "version": "1.0.0",
        "endpoints": {
            "POST /reset": "Start a new episode",
            "POST /step": "Submit an action",
            "GET  /state": "Get current state",
            "GET  /tasks": "List all tasks",
            "GET  /health": "Health check",
        },
        "tasks": VALID_TASK_NAMES,
        "docs": "/docs",
    }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))  # HF Spaces uses port 8000
    print(f"Starting Email Triage Environment server on port {port}...")
    print(f"API docs available at: http://localhost:{port}/docs")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
