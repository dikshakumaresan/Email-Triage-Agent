"""
environment.py
--------------
Core Email Triage Environment implementing the OpenEnv interface.

Three methods the OpenEnv spec requires:
  reset()        -> returns initial observation
  step(action)   -> returns observation, reward, done, info
  state()        -> returns current internal state

Supports all 3 tasks:
  task1_priority_classification  (easy)
  task2_routing_and_reply        (medium)
  task3_thread_analysis          (hard)
"""

from typing import Any, Optional
from pydantic import BaseModel

from tasks import (
    get_task,
    get_observation_for_step,
    list_tasks,
    Task1Observation,
    Task2Observation,
    Task3Observation,
    Task1Action,
    Task2Action,
    Task3Action,
    EmailReward,
)
from graders import grade
from emails import TASK1_EMAILS, TASK2_EMAILS, TASK3_THREADS


# ============================================================
# STEP RESULT — what env.step() returns
# ============================================================

class StepResult(BaseModel):
    observation: Optional[dict] = None   # next email/thread to process
    reward: float = 0.0                  # reward for this step (0.0 - 1.0)
    done: bool = False                   # is the episode over?
    info: dict = {}                      # extra info (feedback, scores breakdown)


# ============================================================
# RESET RESULT — what env.reset() returns
# ============================================================

class ResetResult(BaseModel):
    observation: dict                    # first email/thread to process
    info: dict = {}                      # task description and instructions


# ============================================================
# STATE — what env.state() returns
# ============================================================

class EnvironmentState(BaseModel):
    task_name: str
    current_step: int
    max_steps: int
    total_reward: float
    episode_rewards: list[float]
    done: bool
    emails_processed: int
    emails_remaining: int


# ============================================================
# MAIN ENVIRONMENT CLASS
# ============================================================

class EmailTriageEnv:
    """
    Email Triage Environment.

    An AI agent processes emails one at a time (or threads for task3).
    For each email, the agent classifies priority, routes it, and drafts a reply.
    A grader scores the response and returns a reward between 0.0 and 1.0.

    Usage:
        env = EmailTriageEnv(task_name="task1_priority_classification")
        result = env.reset()
        while not done:
            action = agent.decide(result.observation)
            result = env.step(action)
            done = result.done
    """

    def __init__(self, task_name: str = "task1_priority_classification"):
        """
        Initialize the environment with a specific task.

        Args:
            task_name: One of:
                - "task1_priority_classification"
                - "task2_routing_and_reply"
                - "task3_thread_analysis"
        """
        # Validate task name immediately
        self.task = get_task(task_name)
        self.task_name = task_name

        # Internal state — reset() will initialize these properly
        self._current_step = 0
        self._done = False
        self._total_reward = 0.0
        self._episode_rewards = []
        self._last_reward_breakdown = {}
        self._last_feedback = ""

        # Load items (emails or threads) for this task
        if task_name == "task3_thread_analysis":
            self._items = TASK3_THREADS
        elif task_name == "task2_routing_and_reply":
            self._items = TASK2_EMAILS
        else:
            self._items = TASK1_EMAILS

        self._max_steps = self.task["max_steps"]

    # ----------------------------------------------------------
    # reset() — start a fresh episode
    # ----------------------------------------------------------

    def reset(self) -> ResetResult:
        """
        Reset the environment to its initial state.
        Returns the first email/thread as the initial observation.

        Returns:
            ResetResult with:
              - observation: the first email or thread
              - info: task description and instructions
        """
        # Reset all internal state
        self._current_step = 0
        self._done = False
        self._total_reward = 0.0
        self._episode_rewards = []
        self._last_reward_breakdown = {}
        self._last_feedback = ""

        # Build the first observation
        observation = self._build_observation(step_index=0)

        return ResetResult(
            observation=observation,
            info={
                "task_name": self.task_name,
                "display_name": self.task["display_name"],
                "difficulty": self.task["difficulty"],
                "description": self.task["description"],
                "objective": self.task["objective"],
                "max_steps": self._max_steps,
                "scoring_guide": self.task["scoring_guide"],
                "message": (
                    f"Episode started. You will process {self._max_steps} "
                    f"{'threads' if self.task_name == 'task3_thread_analysis' else 'emails'}. "
                    f"Task: {self.task['display_name']}"
                ),
            },
        )

    # ----------------------------------------------------------
    # step() — agent takes one action
    # ----------------------------------------------------------

    def step(self, action: dict) -> StepResult:
        """
        Process one agent action.

        The agent submits its answer for the current email/thread.
        The grader scores it and the next email/thread is returned.

        Args:
            action: dict with the agent's answer. Fields depend on task:
                Task 1: {"priority": ..., "category": ...}
                Task 2: {"priority": ..., "category": ..., "route_to": ..., "draft_reply": ...}
                Task 3: {"priority": ..., "category": ..., "route_to": ...,
                         "required_actions": [...], "draft_reply": ..., "context_summary": ...}

        Returns:
            StepResult with:
              - observation: next email/thread (or None if done)
              - reward: score for this step (0.0 - 1.0)
              - done: True if episode is over
              - info: score breakdown and feedback
        """
        # Guard: episode already over
        if self._done:
            return StepResult(
                observation=None,
                reward=0.0,
                done=True,
                info={"error": "Episode is already done. Call reset() to start a new episode."},
            )

        # Get the current item (email or thread)
        current_item = self._items[self._current_step]
        ground_truth = current_item["ground_truth"]

        # --- Grade the action ---
        if self.task_name == "task3_thread_analysis":
            reward_obj = grade(
                task_name=self.task_name,
                action=action,
                ground_truth=ground_truth,
                thread=current_item["thread"],
            )
        else:
            reward_obj = grade(
                task_name=self.task_name,
                action=action,
                ground_truth=ground_truth,
            )

        reward = reward_obj.total_reward
        self._total_reward += reward
        self._episode_rewards.append(reward)
        self._last_feedback = reward_obj.feedback
        self._last_reward_breakdown = reward_obj.model_dump()

        # --- Advance step ---
        self._current_step += 1

        # Check if episode is done
        if self._current_step >= self._max_steps:
            self._done = True

        # --- Build next observation (or None if done) ---
        if self._done:
            next_observation = None
        else:
            next_observation = self._build_observation(step_index=self._current_step)

        # Build info dict
        info = {
            "step": self._current_step,
            "reward_breakdown": self._last_reward_breakdown,
            "feedback": self._last_feedback,
            "total_reward_so_far": round(self._total_reward, 3),
            "average_reward_so_far": round(
                self._total_reward / self._current_step, 3
            ),
            "steps_remaining": self._max_steps - self._current_step,
        }

        if self._done:
            info["episode_summary"] = {
                "total_reward": round(self._total_reward, 3),
                "average_reward": round(self._total_reward / self._max_steps, 3),
                "episode_rewards": self._episode_rewards,
                "final_score": round(self._total_reward / self._max_steps, 3),
                "message": "Episode complete! Call reset() to start a new episode.",
            }

        return StepResult(
            observation=next_observation,
            reward=reward,
            done=self._done,
            info=info,
        )

    # ----------------------------------------------------------
    # state() — inspect current environment state
    # ----------------------------------------------------------

    def state(self) -> EnvironmentState:
        """
        Return the current internal state of the environment.
        Useful for debugging and monitoring.

        Returns:
            EnvironmentState with current step, rewards, and progress info.
        """
        items_processed = self._current_step
        items_remaining = self._max_steps - self._current_step

        return EnvironmentState(
            task_name=self.task_name,
            current_step=self._current_step,
            max_steps=self._max_steps,
            total_reward=round(self._total_reward, 3),
            episode_rewards=self._episode_rewards,
            done=self._done,
            emails_processed=items_processed,
            emails_remaining=items_remaining,
        )

    # ----------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------

    def _build_observation(self, step_index: int) -> dict:
        """
        Build the observation dict for a given step index.
        Formats raw email/thread data into the correct observation model.
        """
        item = self._items[step_index]
        items_remaining = self._max_steps - step_index - 1

        if self.task_name == "task1_priority_classification":
            obs = Task1Observation(
                email_id=item["email_id"],
                sender=item["sender"],
                subject=item["subject"],
                body=item["body"],
                timestamp=item["timestamp"],
                emails_remaining=items_remaining,
            )

        elif self.task_name == "task2_routing_and_reply":
            obs = Task2Observation(
                email_id=item["email_id"],
                sender=item["sender"],
                subject=item["subject"],
                body=item["body"],
                timestamp=item["timestamp"],
                emails_remaining=items_remaining,
            )

        elif self.task_name == "task3_thread_analysis":
            obs = Task3Observation(
                thread_id=item["thread_id"],
                subject=item["subject"],
                thread=item["thread"],
                thread_length=len(item["thread"]),
                threads_remaining=items_remaining,
            )

        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        return obs.model_dump()

    def get_task_info(self) -> dict:
        """Return task metadata — useful for agents to understand the task."""
        return {
            "task_name": self.task_name,
            "display_name": self.task["display_name"],
            "difficulty": self.task["difficulty"],
            "description": self.task["description"],
            "objective": self.task["objective"],
            "max_steps": self._max_steps,
            "scoring_guide": self.task["scoring_guide"],
        }

    def list_available_tasks(self) -> list:
        """Return all available tasks."""
        return list_tasks()


# ============================================================
# QUICK TEST — run this file directly to verify the env works
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ENVIRONMENT TESTS")
    print("=" * 60)

    # --- Test Task 1 ---
    print("\n[Task 1 - Full episode simulation]")
    env = EmailTriageEnv(task_name="task1_priority_classification")

    reset_result = env.reset()
    print(f"  Task: {reset_result.info['display_name']}")
    print(f"  First email subject: {reset_result.observation['subject']}")

    done = False
    step = 0
    while not done:
        step += 1
        # Simulate a perfect agent
        obs = reset_result.observation if step == 1 else result.observation
        result = env.step({
            "priority": "critical",
            "category": "technical_incident",
        })
        print(f"  Step {step}: reward={result.reward} done={result.done}")
        done = result.done

    final = result.info.get("episode_summary", {})
    print(f"  Final score: {final.get('final_score', 'N/A')}")

    # --- Test state() ---
    print("\n[state() output]")
    state = env.state()
    print(f"  {state.model_dump()}")

    # --- Test Task 2 ---
    print("\n[Task 2 - One step test]")
    env2 = EmailTriageEnv(task_name="task2_routing_and_reply")
    reset2 = env2.reset()
    print(f"  Task: {reset2.info['display_name']}")

    result2 = env2.step({
        "priority": "high",
        "category": "billing",
        "route_to": "finance_team",
        "draft_reply": (
            "Thank you for reaching out about invoice #4521. "
            "Our finance team will follow up with you within 24 hours."
        ),
    })
    print(f"  Step 1 reward: {result2.reward}")
    print(f"  Feedback: {result2.info['feedback']}")

    print("\n" + "=" * 60)
    print("All environment tests passed!")
