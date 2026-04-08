"""
server/app.py
-------------
Corrected FastAPI web server for OpenEnv multi-mode deployment.
"""

import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the parent directory to sys.path so we can find environment.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# ============================================================

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
            detail=f"Unknown task '{task_name}'. Valid tasks: {VALID_TASK_NAMES}",
        )
    if task_name not in _envs:
        _envs[task_name] = EmailTriageEnv(task_name=task_name)
    return _envs[task_name]


# ============================================================
# MODELS
# ============================================================

class ResetRequest(BaseModel):
    task_name: Optional[str] = DEFAULT_TASK

class StepRequest(BaseModel):
    task_name: Optional[str] = DEFAULT_TASK
    action: dict

class ResetResponse(BaseModel):
    observation: dict
    info: dict

class StepResponse(BaseModel):
    observation: Optional[dict]
    reward: float
    done: bool
    info: dict

class StateResponse(BaseModel):
    task_name: str
    current_step: int
    max_steps: int
    total_reward: float
    episode_rewards: list
    done: bool
    emails_processed: int
    emails_remaining: int

class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str
    available_tasks: list

class TasksResponse(BaseModel):
    tasks: list


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        environment="email-triage-env",
        version="1.0.0",
        available_tasks=VALID_TASK_NAMES,
    )

@app.get("/tasks", response_model=TasksResponse)
def list_tasks():
    env = get_env(DEFAULT_TASK)
    return TasksResponse(tasks=env.list_available_tasks())

@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = None):
    if request is None:
        request = ResetRequest()
    task_name = request.task_name or DEFAULT_TASK
    env = get_env(task_name)
    result: ResetResult = env.reset()
    return ResetResponse(observation=result.observation, info=result.info)

@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
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
    return {
        "name": "Email Triage Environment",
        "version": "1.0.0",
        "tasks": VALID_TASK_NAMES,
    }

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860)) 
    # CRITICAL: Use server.app:app because the file is in the server folder
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)
