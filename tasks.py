"""
tasks.py
--------
Defines the 3 tasks for the Email Triage Environment.

Task 1 — Easy:   Priority Classification
Task 2 — Medium: Priority + Routing + Reply Draft
Task 3 — Hard:   Multi-Email Thread with Conflicting Signals

Each task has:
  - name / description
  - what inputs the agent sees (observation fields)
  - what outputs the agent must produce (action fields)
  - the emails/threads it uses
  - max steps allowed
  - difficulty level
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from emails import (
    TASK1_EMAILS,
    TASK2_EMAILS,
    TASK3_THREADS,
    VALID_PRIORITIES,
    VALID_CATEGORIES,
    VALID_ROUTES,
)


# ============================================================
# OBSERVATION MODELS
# These are what the AGENT SEES at each step
# ============================================================

class Task1Observation(BaseModel):
    """What the agent sees in Task 1 — a single email."""
    email_id: str = Field(description="Unique ID of the email")
    sender: str = Field(description="Email sender address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Full email body text")
    timestamp: str = Field(description="When the email was received")
    emails_remaining: int = Field(description="How many emails are left in the inbox")
    valid_priorities: List[str] = Field(
        default=VALID_PRIORITIES,
        description="List of valid priority values you can use"
    )
    valid_categories: List[str] = Field(
        default=VALID_CATEGORIES,
        description="List of valid category values you can use"
    )


class Task2Observation(BaseModel):
    """What the agent sees in Task 2 — email + routing context."""
    email_id: str = Field(description="Unique ID of the email")
    sender: str = Field(description="Email sender address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Full email body text")
    timestamp: str = Field(description="When the email was received")
    emails_remaining: int = Field(description="How many emails are left in the inbox")
    valid_priorities: List[str] = Field(default=VALID_PRIORITIES)
    valid_categories: List[str] = Field(default=VALID_CATEGORIES)
    valid_routes: List[str] = Field(
        default=VALID_ROUTES,
        description="List of valid teams you can route this email to"
    )


class Task3Observation(BaseModel):
    """What the agent sees in Task 3 — a full email thread."""
    thread_id: str = Field(description="Unique ID of the email thread")
    subject: str = Field(description="Thread subject")
    thread: List[dict] = Field(description="Full list of emails in this thread, oldest first")
    thread_length: int = Field(description="Number of emails in the thread")
    threads_remaining: int = Field(description="How many threads are left to process")
    valid_priorities: List[str] = Field(default=VALID_PRIORITIES)
    valid_categories: List[str] = Field(default=VALID_CATEGORIES)
    valid_routes: List[str] = Field(default=VALID_ROUTES)


# ============================================================
# ACTION MODELS
# These are what the AGENT MUST OUTPUT at each step
# ============================================================

class Task1Action(BaseModel):
    """What the agent must output for Task 1."""
    priority: str = Field(
        description=f"Email priority. Must be one of: {VALID_PRIORITIES}"
    )
    category: str = Field(
        description=f"Email category. Must be one of: {VALID_CATEGORIES}"
    )


class Task2Action(BaseModel):
    """What the agent must output for Task 2."""
    priority: str = Field(
        description=f"Email priority. Must be one of: {VALID_PRIORITIES}"
    )
    category: str = Field(
        description=f"Email category. Must be one of: {VALID_CATEGORIES}"
    )
    route_to: str = Field(
        description=f"Which team to route this email to. Must be one of: {VALID_ROUTES}"
    )
    draft_reply: str = Field(
        description="A professional reply to send to the email sender. Minimum 20 words."
    )


class Task3Action(BaseModel):
    """What the agent must output for Task 3."""
    priority: str = Field(
        description=f"Thread priority. Must be one of: {VALID_PRIORITIES}"
    )
    category: str = Field(
        description=f"Thread category. Must be one of: {VALID_CATEGORIES}"
    )
    route_to: str = Field(
        description=f"Which team to route this to. Must be one of: {VALID_ROUTES}"
    )
    required_actions: List[str] = Field(
        description="List of actions that must be taken based on the thread context."
    )
    draft_reply: str = Field(
        description="A professional reply addressing ALL points raised in the thread. Minimum 40 words."
    )
    context_summary: str = Field(
        description="Brief summary of what happened in this thread — shows you read all emails."
    )


# ============================================================
# REWARD MODEL
# Returned after every step — shows partial scores
# ============================================================

class EmailReward(BaseModel):
    """Reward breakdown after each step."""
    priority_score: float = Field(description="Score for priority classification (0.0-1.0)")
    category_score: float = Field(description="Score for category classification (0.0-1.0)")
    routing_score: float = Field(default=0.0, description="Score for routing decision (0.0-1.0)")
    reply_score: float = Field(default=0.0, description="Score for reply quality (0.0-1.0)")
    actions_score: float = Field(default=0.0, description="Score for required actions identified (0.0-1.0)")
    context_score: float = Field(default=0.0, description="Score for thread context understanding (0.0-1.0)")
    total_reward: float = Field(description="Final weighted reward (0.0-1.0)")
    feedback: str = Field(description="Human-readable feedback on what was right/wrong")


# ============================================================
# TASK DEFINITIONS
# ============================================================

TASK_DEFINITIONS = {

    # ----------------------------------------------------------
    "task1_priority_classification": {
        "name": "task1_priority_classification",
        "display_name": "Task 1: Priority Classification",
        "difficulty": "easy",
        "description": (
            "Read each email and classify it by priority level and category. "
            "You will process 5 emails one at a time. "
            "For each email, output the correct priority (critical/high/medium/low) "
            "and category."
        ),
        "objective": (
            "Correctly classify the priority and category of each email. "
            "Score is based on accuracy of your classifications."
        ),
        "max_steps": 5,               # one step per email
        "emails": TASK1_EMAILS,
        "observation_model": Task1Observation,
        "action_model": Task1Action,
        # Reward weights — must sum to 1.0
        "reward_weights": {
            "priority": 0.5,
            "category": 0.5,
        },
        "scoring_guide": {
            "perfect": "Both priority and category correct → 1.0",
            "partial": "Only priority correct → 0.5 | Only category correct → 0.5",
            "wrong": "Both wrong → 0.0",
        },
    },

    # ----------------------------------------------------------
    "task2_routing_and_reply": {
        "name": "task2_routing_and_reply",
        "display_name": "Task 2: Priority + Routing + Reply",
        "difficulty": "medium",
        "description": (
            "Read each email and classify its priority, route it to the correct team, "
            "and draft a professional reply. "
            "You will process 5 emails one at a time."
        ),
        "objective": (
            "Correctly classify priority, route to the right team, "
            "and write a relevant, professional reply for each email."
        ),
        "max_steps": 5,
        "emails": TASK2_EMAILS,
        "observation_model": Task2Observation,
        "action_model": Task2Action,
        "reward_weights": {
            "priority": 0.25,
            "category": 0.15,
            "routing": 0.30,
            "reply": 0.30,
        },
        "scoring_guide": {
            "perfect": "All fields correct + reply addresses key topics → 1.0",
            "partial": "Each component scored independently and averaged",
            "wrong": "All fields wrong → 0.0",
        },
    },

    # ----------------------------------------------------------
    "task3_thread_analysis": {
        "name": "task3_thread_analysis",
        "display_name": "Task 3: Multi-Email Thread Analysis",
        "difficulty": "hard",
        "description": (
            "Read full email threads (multiple emails back and forth) and make "
            "nuanced decisions. Threads may contain conflicting information — "
            "you must understand the full context before responding. "
            "You will process 3 threads."
        ),
        "objective": (
            "Correctly assess priority, route the thread, identify all required actions, "
            "write a reply that addresses ALL points, and demonstrate you understood "
            "the full thread context."
        ),
        "max_steps": 3,               # one step per thread
        "threads": TASK3_THREADS,
        "observation_model": Task3Observation,
        "action_model": Task3Action,
        "reward_weights": {
            "priority": 0.15,
            "category": 0.10,
            "routing": 0.20,
            "reply": 0.25,
            "actions": 0.20,
            "context": 0.10,
        },
        "scoring_guide": {
            "perfect": "All components correct + full context demonstrated → 1.0",
            "partial": "Each component scored independently — partial credit given",
            "wrong": "Ignored thread history or missed key actions → low score",
        },
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_task(task_name: str) -> dict:
    """Get a task definition by name. Raises error if not found."""
    if task_name not in TASK_DEFINITIONS:
        available = list(TASK_DEFINITIONS.keys())
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available}")
    return TASK_DEFINITIONS[task_name]


def list_tasks() -> List[dict]:
    """Return a summary of all available tasks."""
    return [
        {
            "name": t["name"],
            "display_name": t["display_name"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "max_steps": t["max_steps"],
        }
        for t in TASK_DEFINITIONS.values()
    ]


def get_observation_for_step(task_name: str, step_index: int) -> dict:
    """
    Get the raw email/thread data for a given step index.
    step_index is 0-based internally.
    """
    task = get_task(task_name)

    if task_name == "task3_thread_analysis":
        items = task["threads"]
    else:
        items = task["emails"]

    if step_index >= len(items):
        raise IndexError(
            f"Step index {step_index} out of range for task '{task_name}' "
            f"which has {len(items)} items."
        )

    return items[step_index]
