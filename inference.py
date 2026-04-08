"""
inference.py
------------
Baseline inference script for the Email Triage Environment.

Runs an LLM agent against all 3 tasks and emits structured logs
in the exact format required by the OpenEnv hackathon spec:

  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

MANDATORY ENVIRONMENT VARIABLES:
  API_BASE_URL   The API endpoint for the LLM
  MODEL_NAME     The model identifier to use
  HF_TOKEN       Your Hugging Face / API key

Usage:
  python inference.py

  # With custom settings:
  API_BASE_URL=https://router.huggingface.co/v1 \
  MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  HF_TOKEN=your_token \
  python inference.py
"""

import os
import json
import textwrap
import requests
from typing import Optional
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

SERVER_URL   = os.getenv("SERVER_URL",   "http://localhost:7860")
BENCHMARK    = "email-triage-env"

# Max steps safety cap (env already enforces its own limits)
MAX_STEPS    = 10

# Score threshold to count episode as success
SUCCESS_THRESHOLD = 0.5

# All 3 tasks to run
ALL_TASKS = [
    "task1_priority_classification",
    "task2_routing_and_reply",
    "task3_thread_analysis",
]


# ============================================================
# LOGGING — exact format required by hackathon spec
# ============================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Collapse action to single line for log
    action_str = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ============================================================
# ENVIRONMENT CLIENT
# Talks to the FastAPI server via HTTP
# ============================================================

class EnvClient:
    """Simple HTTP client for the Email Triage Environment server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_name: str) -> dict:
        """Call POST /reset and return the result."""
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_name": task_name},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, task_name: str, action: dict) -> dict:
        """Call POST /step and return the result."""
        resp = requests.post(
            f"{self.base_url}/step",
            json={"task_name": task_name, "action": action},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def state(self, task_name: str) -> dict:
        """Call GET /state and return the result."""
        resp = requests.get(
            f"{self.base_url}/state",
            params={"task_name": task_name},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


# ============================================================
# SYSTEM PROMPTS — one per task
# Tells the LLM exactly what to do and what format to output
# ============================================================

TASK1_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert email triage assistant.
    Your job is to read each email and classify it.

    You must respond with ONLY a valid JSON object — no explanation, no markdown, no extra text.

    Required fields:
      "priority"  : one of ["critical", "high", "medium", "low"]
      "category"  : one of [
          "technical_incident", "spam", "hr_admin", "meeting_scheduling",
          "security", "billing", "technical_support", "business_development",
          "hr_complaint", "press_media", "account_management",
          "billing_escalation", "recruitment", "other"
      ]

    Example response:
    {"priority": "critical", "category": "technical_incident"}

    Rules:
    - critical = needs immediate action (system down, security breach, legal threat)
    - high     = important, needs action today
    - medium   = needs action this week
    - low      = informational, no action needed or spam
""").strip()


TASK2_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert email triage assistant.
    Your job is to read each email, classify it, route it to the right team,
    and draft a professional reply.

    You must respond with ONLY a valid JSON object — no explanation, no markdown, no extra text.

    Required fields:
      "priority"    : one of ["critical", "high", "medium", "low"]
      "category"    : one of [
          "technical_incident", "spam", "hr_admin", "meeting_scheduling",
          "security", "billing", "technical_support", "business_development",
          "hr_complaint", "press_media", "account_management",
          "billing_escalation", "recruitment", "other"
      ]
      "route_to"    : one of [
          "it_support", "finance_team", "hr_team", "sales_team",
          "pr_legal_team", "account_management_team", "hr_manager",
          "no_action_needed", "other"
      ]
      "draft_reply" : a professional reply to the sender (minimum 20 words)

    Example response:
    {
      "priority": "high",
      "category": "billing",
      "route_to": "finance_team",
      "draft_reply": "Thank you for reaching out regarding your invoice. Our finance team will review this and follow up with you within 24 hours."
    }

    Rules for routing:
    - it_support            -> technical/login/password issues
    - finance_team          -> billing, invoices, payments, refunds
    - hr_team               -> harassment, complaints, HR policy
    - sales_team            -> partnerships, business development
    - pr_legal_team         -> press, media, legal threats
    - account_management_team -> subscription, account changes
    - hr_manager            -> job offers, recruitment, salary
    - no_action_needed      -> spam, newsletters

    Rules for draft_reply:
    - Be professional and courteous
    - Acknowledge the sender's concern
    - State what action will be taken
    - Give a timeframe when possible
    - Minimum 20 words
""").strip()


TASK3_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert email triage assistant handling complex email threads.
    You must read the ENTIRE thread carefully — emails may contradict each other.
    The LATEST email is what matters most, but earlier emails provide critical context.

    You must respond with ONLY a valid JSON object — no explanation, no markdown, no extra text.

    Required fields:
      "priority"        : one of ["critical", "high", "medium", "low"]
      "category"        : one of [
          "technical_incident", "spam", "hr_admin", "meeting_scheduling",
          "security", "billing", "technical_support", "business_development",
          "hr_complaint", "press_media", "account_management",
          "billing_escalation", "recruitment", "other"
      ]
      "route_to"        : one of [
          "it_support", "finance_team", "hr_team", "sales_team",
          "pr_legal_team", "account_management_team", "hr_manager",
          "no_action_needed", "other"
      ]
      "required_actions": list of strings describing actions that MUST be taken
      "draft_reply"     : professional reply addressing ALL points in the thread (min 40 words)
      "context_summary" : brief summary of what happened across the full thread (min 20 words)

    Example response:
    {
      "priority": "high",
      "category": "account_management",
      "route_to": "account_management_team",
      "required_actions": ["reverse_cancellation", "process_upgrade"],
      "draft_reply": "Thank you for reaching out. We understand you initially requested cancellation but have since changed your mind. We will immediately reverse the cancellation and process your upgrade to Enterprise. Our team will confirm both changes within the hour.",
      "context_summary": "Client requested cancellation on Jan 12, which was processed. They then changed their mind on Jan 15 and also want to upgrade to Enterprise plan. Both actions need to be taken."
    }

    Important rules:
    - READ ALL EMAILS in the thread before deciding
    - The situation may have changed from the first email to the last
    - required_actions should list every concrete action that needs to happen
    - draft_reply must address EVERY open question or concern in the thread
    - context_summary must show you understood the full history
""").strip()

SYSTEM_PROMPTS = {
    "task1_priority_classification": TASK1_SYSTEM_PROMPT,
    "task2_routing_and_reply":       TASK2_SYSTEM_PROMPT,
    "task3_thread_analysis":         TASK3_SYSTEM_PROMPT,
}


# ============================================================
# USER PROMPT BUILDER
# Converts the observation dict into a prompt the LLM can read
# ============================================================

def build_user_prompt(task_name: str, observation: dict) -> str:
    """Build a clear user prompt from the current observation."""

    if task_name == "task3_thread_analysis":
        # Format the full email thread
        thread_lines = []
        for i, email in enumerate(observation.get("thread", []), 1):
            thread_lines.append(
                f"Email {i} ({email.get('timestamp', 'unknown time')}):\n"
                f"  From: {email.get('from', 'unknown')}\n"
                f"  Message: {email.get('body', '')}"
            )
        thread_text = "\n\n".join(thread_lines)

        return textwrap.dedent(f"""
            Thread Subject: {observation.get('subject', '')}
            Number of emails in thread: {observation.get('thread_length', 0)}
            Threads remaining after this: {observation.get('threads_remaining', 0)}

            --- FULL THREAD ---
            {thread_text}
            -------------------

            Read all {observation.get('thread_length', 0)} emails carefully.
            Respond with your JSON assessment now.
        """).strip()

    else:
        # Format a single email
        return textwrap.dedent(f"""
            From:    {observation.get('sender', '')}
            Subject: {observation.get('subject', '')}
            Time:    {observation.get('timestamp', '')}

            --- EMAIL BODY ---
            {observation.get('body', '')}
            ------------------

            Emails remaining after this: {observation.get('emails_remaining', 0)}

            Respond with your JSON assessment now.
        """).strip()


# ============================================================
# LLM CALL
# ============================================================

def call_llm(client: OpenAI, task_name: str, observation: dict) -> tuple[dict, str]:
    """
    Call the LLM with the current observation.

    Returns:
        (action_dict, raw_response_string)
    """
    system_prompt = SYSTEM_PROMPTS[task_name]
    user_prompt   = build_user_prompt(task_name, observation)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,   # Low temperature for more consistent outputs
            max_tokens=500,
            stream=False,
        )

        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if LLM added them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

        # Parse JSON
        action = json.loads(raw)
        return action, raw

    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e} | Raw: {raw[:200]}", flush=True)
        # Return a safe fallback action
        return _fallback_action(task_name), f"JSON_PARSE_ERROR: {e}"

    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return _fallback_action(task_name), f"LLM_ERROR: {e}"


def _fallback_action(task_name: str) -> dict:
    """Return a minimal valid action when LLM fails."""
    base = {"priority": "medium", "category": "other"}
    if task_name == "task2_routing_and_reply":
        base.update({"route_to": "other", "draft_reply": "Thank you for your email. We will get back to you shortly."})
    elif task_name == "task3_thread_analysis":
        base.update({
            "route_to": "other",
            "required_actions": ["review_thread"],
            "draft_reply": "Thank you for your email thread. We are reviewing all points and will respond shortly.",
            "context_summary": "Thread reviewed. Action required.",
        })
    return base


# ============================================================
# RUN ONE TASK EPISODE
# ============================================================

def run_task(client: OpenAI, env_client: EnvClient, task_name: str) -> float:
    """
    Run one full episode for a given task.

    Returns:
        final_score (float between 0.0 and 1.0)
    """
    rewards      = []
    steps_taken  = 0
    success      = False
    final_score  = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # --- Reset environment ---
        reset_result = env_client.reset(task_name)
        observation  = reset_result["observation"]
        done         = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # --- Agent decides action ---
            action, raw_action = call_llm(client, task_name, observation)

            # --- Submit to environment ---
            try:
                step_result = env_client.step(task_name, action)
                reward      = step_result.get("reward", 0.0)
                done        = step_result.get("done", False)
                observation = step_result.get("observation")
                error       = step_result.get("info", {}).get("error")
            except Exception as e:
                reward = 0.0
                done   = True
                error  = str(e)
                print(f"[DEBUG] Step error: {e}", flush=True)

            rewards.append(reward)
            steps_taken = step

            # Log in required format
            log_step(
                step=step,
                action=json.dumps(action, separators=(",", ":")),
                reward=reward,
                done=done,
                error=error if "error" in step_result.get("info", {}) else None,
            )

            if done:
                break

        # --- Calculate final score ---
        if rewards:
            final_score = sum(rewards) / len(rewards)
            final_score = round(min(max(final_score, 0.0), 1.0), 3)

        success = final_score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        success = False

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards,
        )

    return final_score


# ============================================================
# MAIN — run all 3 tasks
# ============================================================

def main():
    print("=" * 60, flush=True)
    print("Email Triage Environment — Baseline Inference", flush=True)
    print(f"Model:      {MODEL_NAME}", flush=True)
    print(f"Server:     {SERVER_URL}", flush=True)
    print(f"API Base:   {API_BASE_URL}", flush=True)
    print("=" * 60, flush=True)

    # --- Validate config ---
    if not HF_TOKEN:
        print("[WARNING] HF_TOKEN not set. LLM calls may fail.", flush=True)

    # --- Set up clients ---
    client     = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    env_client = EnvClient(base_url=SERVER_URL)

    # --- Check server is alive ---
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=10)
        resp.raise_for_status()
        print(f"[INFO] Server is live: {resp.json()}", flush=True)
    except Exception as e:
        print(f"[ERROR] Cannot reach server at {SERVER_URL}: {e}", flush=True)
        print("[ERROR] Make sure the server is running: python server.py", flush=True)
        return

    print("", flush=True)

    # --- Run all 3 tasks ---
    all_scores = {}

    for task_name in ALL_TASKS:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Running: {task_name}", flush=True)
        print("=" * 60, flush=True)

        score = run_task(client, env_client, task_name)
        all_scores[task_name] = score

        print(f"\n[RESULT] {task_name}: {score:.3f}", flush=True)

    # --- Final summary ---
    print(f"\n{'=' * 60}", flush=True)
    print("FINAL SCORES", flush=True)
    print("=" * 60, flush=True)
    for task_name, score in all_scores.items():
        difficulty = {
            "task1_priority_classification": "easy",
            "task2_routing_and_reply":       "medium",
            "task3_thread_analysis":         "hard",
        }[task_name]
        status = "✓ PASS" if score >= SUCCESS_THRESHOLD else "✗ FAIL"
        print(f"  {status}  {task_name} ({difficulty}): {score:.3f}", flush=True)

    overall = sum(all_scores.values()) / len(all_scores)
    print(f"\n  Overall average score: {overall:.3f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
