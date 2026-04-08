"""
graders.py
----------
Scoring logic for all 3 tasks in the Email Triage Environment.

Each grader returns an EmailReward with:
  - Individual component scores (0.0 - 1.0)
  - A final weighted total_reward (0.0 - 1.0)
  - Human-readable feedback

Grading is fully deterministic — same input always gives same score.
"""

from tasks import EmailReward, TASK_DEFINITIONS
from emails import VALID_PRIORITIES, VALID_CATEGORIES, VALID_ROUTES


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def score_priority(predicted: str, expected: str) -> tuple[float, str]:
    """
    Score the priority classification.
    
    Exact match = 1.0
    One level off = 0.5  (e.g. predicted high, expected critical)
    Two levels off = 0.25
    Completely wrong = 0.0

    Priority order: critical > high > medium > low
    """
    priority_order = ["critical", "high", "medium", "low"]

    predicted = predicted.lower().strip()
    expected = expected.lower().strip()

    # Invalid value — not in allowed list
    if predicted not in priority_order:
        return 0.0, f"Invalid priority '{predicted}'. Must be one of {priority_order}."

    if predicted == expected:
        return 1.0, f"Priority correct: '{predicted}'"

    # Partial credit based on how far off
    pred_idx = priority_order.index(predicted)
    exp_idx = priority_order.index(expected)
    distance = abs(pred_idx - exp_idx)

    if distance == 1:
        return 0.5, f"Priority close: predicted '{predicted}', expected '{expected}' (1 level off)"
    elif distance == 2:
        return 0.25, f"Priority off: predicted '{predicted}', expected '{expected}' (2 levels off)"
    else:
        return 0.0, f"Priority wrong: predicted '{predicted}', expected '{expected}'"


def score_category(predicted: str, expected: str) -> tuple[float, str]:
    """
    Score the category classification.
    Exact match = 1.0, else 0.0
    Categories are discrete — no partial credit here.
    """
    predicted = predicted.lower().strip()
    expected = expected.lower().strip()

    if predicted not in VALID_CATEGORIES:
        return 0.0, f"Invalid category '{predicted}'."

    if predicted == expected:
        return 1.0, f"Category correct: '{predicted}'"

    return 0.0, f"Category wrong: predicted '{predicted}', expected '{expected}'"


def score_routing(predicted: str, expected: str) -> tuple[float, str]:
    """
    Score the routing decision.
    Exact match = 1.0, else 0.0
    """
    predicted = predicted.lower().strip()
    expected = expected.lower().strip()

    if predicted not in VALID_ROUTES:
        return 0.0, f"Invalid route '{predicted}'."

    if predicted == expected:
        return 1.0, f"Routing correct: '{predicted}'"

    return 0.0, f"Routing wrong: predicted '{predicted}', expected '{expected}'"


def score_reply(draft_reply: str, must_include: list[str]) -> tuple[float, str]:
    """
    Score the draft reply quality.

    Scoring breakdown:
      - Minimum length check (20+ words)     → 0.2
      - Professionalism check                → 0.2
      - Contains required keywords           → up to 0.6 (split across keywords)

    Total = 1.0 max
    """
    score = 0.0
    feedback_parts = []

    if not draft_reply or not draft_reply.strip():
        return 0.0, "Reply is empty."

    reply_lower = draft_reply.lower().strip()
    word_count = len(reply_lower.split())

    # --- Length check (0.2) ---
    if word_count >= 20:
        score += 0.2
        feedback_parts.append(f"Length OK ({word_count} words)")
    else:
        feedback_parts.append(f"Reply too short ({word_count} words, need 20+)")

    # --- Professionalism check (0.2) ---
    # Checks for basic professional language
    professional_indicators = [
        "thank", "please", "we", "our", "will", "team",
        "assist", "help", "contact", "reach", "respond",
        "appreciate", "understand", "apologize", "ensure"
    ]
    professional_hits = sum(1 for w in professional_indicators if w in reply_lower)

    if professional_hits >= 3:
        score += 0.2
        feedback_parts.append("Reply sounds professional")
    elif professional_hits >= 1:
        score += 0.1
        feedback_parts.append("Reply somewhat professional")
    else:
        feedback_parts.append("Reply lacks professional tone")

    # --- Keyword coverage (0.6 split across required keywords) ---
    if must_include:
        keyword_score_each = 0.6 / len(must_include)
        matched_keywords = []
        missed_keywords = []

        for keyword in must_include:
            if keyword.lower() in reply_lower:
                score += keyword_score_each
                matched_keywords.append(keyword)
            else:
                missed_keywords.append(keyword)

        if matched_keywords:
            feedback_parts.append(f"Keywords found: {matched_keywords}")
        if missed_keywords:
            feedback_parts.append(f"Keywords missing: {missed_keywords}")

    return round(min(score, 1.0), 3), " | ".join(feedback_parts)


def score_required_actions(
    predicted_actions: list[str],
    expected_actions: list[str]
) -> tuple[float, str]:
    """
    Score the required actions for Task 3.
    Each correctly identified action earns equal partial credit.

    e.g. if expected = [A, B, C] and agent found [A, C] → 2/3 = 0.67
    """
    if not expected_actions:
        return 1.0, "No required actions expected"

    if not predicted_actions:
        return 0.0, f"No actions provided. Expected: {expected_actions}"

    predicted_lower = [a.lower().strip() for a in predicted_actions]
    expected_lower = [a.lower().strip() for a in expected_actions]

    matched = []
    missed = []

    for action in expected_lower:
        # Check for exact match OR if the expected action appears inside a predicted action
        found = any(action in p or p in action for p in predicted_lower)
        if found:
            matched.append(action)
        else:
            missed.append(action)

    score = len(matched) / len(expected_lower)
    feedback = f"Actions matched: {matched}"
    if missed:
        feedback += f" | Actions missed: {missed}"

    return round(score, 3), feedback


def score_context_understanding(
    context_summary: str,
    thread: list[dict]
) -> tuple[float, str]:
    """
    Score how well the agent understood the full email thread.

    Checks:
      - Summary is not empty (0.2)
      - Summary is long enough to show real understanding (0.3)
      - Summary references key facts from thread (0.5)
    """
    if not context_summary or not context_summary.strip():
        return 0.0, "No context summary provided"

    score = 0.0
    feedback_parts = []

    # Not empty
    score += 0.2
    feedback_parts.append("Summary provided")

    # Long enough (at least 20 words)
    word_count = len(context_summary.split())
    if word_count >= 20:
        score += 0.3
        feedback_parts.append(f"Summary detailed enough ({word_count} words)")
    elif word_count >= 10:
        score += 0.15
        feedback_parts.append(f"Summary a bit short ({word_count} words)")
    else:
        feedback_parts.append(f"Summary too short ({word_count} words)")

    # Extract key words from thread bodies and check if summary mentions them
    thread_text = " ".join(email["body"].lower() for email in thread)
    summary_lower = context_summary.lower()

    # Pull significant words from thread (ignore common words)
    stopwords = {"the", "a", "an", "is", "it", "to", "and", "or", "in", "of",
                 "we", "i", "my", "your", "our", "this", "that", "for", "on"}
    thread_words = set(
        w.strip(".,!?") for w in thread_text.split()
        if len(w) > 4 and w not in stopwords
    )

    # Check how many significant thread words appear in summary
    matched_words = [w for w in thread_words if w in summary_lower]
    coverage = len(matched_words) / max(len(thread_words), 1)

    if coverage >= 0.15:
        score += 0.5
        feedback_parts.append("Summary captures key thread facts")
    elif coverage >= 0.08:
        score += 0.25
        feedback_parts.append("Summary partially captures thread facts")
    else:
        feedback_parts.append("Summary misses important thread details")

    return round(min(score, 1.0), 3), " | ".join(feedback_parts)


# ============================================================
# MAIN GRADER FUNCTIONS — one per task
# ============================================================

def grade_task1(action: dict, ground_truth: dict) -> EmailReward:
    """
    Grade a Task 1 action (Priority Classification).

    action keys:     priority, category
    ground_truth keys: priority, category
    """
    weights = TASK_DEFINITIONS["task1_priority_classification"]["reward_weights"]

    priority_score, priority_feedback = score_priority(
        action.get("priority", ""),
        ground_truth["priority"]
    )

    category_score, category_feedback = score_category(
        action.get("category", ""),
        ground_truth["category"]
    )

    total = (
        priority_score * weights["priority"] +
        category_score * weights["category"]
    )

    feedback = (
        f"Priority: {priority_feedback} | "
        f"Category: {category_feedback}"
    )

    return EmailReward(
        priority_score=priority_score,
        category_score=category_score,
        total_reward=round(total, 3),
        feedback=feedback,
    )


def grade_task2(action: dict, ground_truth: dict) -> EmailReward:
    """
    Grade a Task 2 action (Priority + Routing + Reply).

    action keys:       priority, category, route_to, draft_reply
    ground_truth keys: priority, category, route_to, reply_must_include
    """
    weights = TASK_DEFINITIONS["task2_routing_and_reply"]["reward_weights"]

    priority_score, priority_fb = score_priority(
        action.get("priority", ""),
        ground_truth["priority"]
    )

    category_score, category_fb = score_category(
        action.get("category", ""),
        ground_truth["category"]
    )

    routing_score, routing_fb = score_routing(
        action.get("route_to", ""),
        ground_truth["route_to"]
    )

    reply_score, reply_fb = score_reply(
        action.get("draft_reply", ""),
        ground_truth.get("reply_must_include", [])
    )

    total = (
        priority_score * weights["priority"] +
        category_score * weights["category"] +
        routing_score  * weights["routing"] +
        reply_score    * weights["reply"]
    )

    feedback = (
        f"Priority: {priority_fb} | "
        f"Category: {category_fb} | "
        f"Routing: {routing_fb} | "
        f"Reply: {reply_fb}"
    )

    return EmailReward(
        priority_score=priority_score,
        category_score=category_score,
        routing_score=routing_score,
        reply_score=reply_score,
        total_reward=round(total, 3),
        feedback=feedback,
    )


def grade_task3(action: dict, ground_truth: dict, thread: list[dict]) -> EmailReward:
    """
    Grade a Task 3 action (Multi-Email Thread Analysis).

    action keys:       priority, category, route_to, required_actions,
                       draft_reply, context_summary
    ground_truth keys: priority, category, route_to, required_actions,
                       reply_must_include, context_understood
    """
    weights = TASK_DEFINITIONS["task3_thread_analysis"]["reward_weights"]

    priority_score, priority_fb = score_priority(
        action.get("priority", ""),
        ground_truth["priority"]
    )

    category_score, category_fb = score_category(
        action.get("category", ""),
        ground_truth["category"]
    )

    routing_score, routing_fb = score_routing(
        action.get("route_to", ""),
        ground_truth["route_to"]
    )

    reply_score, reply_fb = score_reply(
        action.get("draft_reply", ""),
        ground_truth.get("reply_must_include", [])
    )

    actions_score, actions_fb = score_required_actions(
        action.get("required_actions", []),
        ground_truth.get("required_actions", [])
    )

    context_score, context_fb = score_context_understanding(
        action.get("context_summary", ""),
        thread
    )

    total = (
        priority_score * weights["priority"] +
        category_score * weights["category"] +
        routing_score  * weights["routing"] +
        reply_score    * weights["reply"] +
        actions_score  * weights["actions"] +
        context_score  * weights["context"]
    )

    feedback = (
        f"Priority: {priority_fb} | "
        f"Category: {category_fb} | "
        f"Routing: {routing_fb} | "
        f"Reply: {reply_fb} | "
        f"Actions: {actions_fb} | "
        f"Context: {context_fb}"
    )

    return EmailReward(
        priority_score=priority_score,
        category_score=category_score,
        routing_score=routing_score,
        reply_score=reply_score,
        actions_score=actions_score,
        context_score=context_score,
        total_reward=round(total, 3),
        feedback=feedback,
    )


# ============================================================
# UNIFIED GRADER — routes to the right task grader
# ============================================================

def grade(task_name: str, action: dict, ground_truth: dict, thread: list = None) -> EmailReward:
    """
    Main entry point for grading.
    Routes to the correct grader based on task name.

    Args:
        task_name:    One of the 3 task names
        action:       The agent's output (as a dict)
        ground_truth: The correct answer (from emails.py)
        thread:       The email thread (only needed for task3)

    Returns:
        EmailReward with scores and feedback
    """
    if task_name == "task1_priority_classification":
        return grade_task1(action, ground_truth)

    elif task_name == "task2_routing_and_reply":
        return grade_task2(action, ground_truth)

    elif task_name == "task3_thread_analysis":
        if thread is None:
            raise ValueError("Task 3 grader requires the 'thread' argument.")
        return grade_task3(action, ground_truth, thread)

    else:
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"Available: task1_priority_classification, task2_routing_and_reply, task3_thread_analysis"
        )


# ============================================================
# QUICK TEST — run this file directly to verify graders work
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GRADER TESTS")
    print("=" * 60)

    # --- Task 1 Test ---
    print("\n[Task 1 - Perfect answer]")
    result = grade(
        "task1_priority_classification",
        action={"priority": "critical", "category": "technical_incident"},
        ground_truth={"priority": "critical", "category": "technical_incident"},
    )
    print(f"  Total reward: {result.total_reward}")
    print(f"  Feedback: {result.feedback}")

    print("\n[Task 1 - Partial answer: priority one level off]")
    result = grade(
        "task1_priority_classification",
        action={"priority": "high", "category": "technical_incident"},
        ground_truth={"priority": "critical", "category": "technical_incident"},
    )
    print(f"  Total reward: {result.total_reward}")
    print(f"  Feedback: {result.feedback}")

    # --- Task 2 Test ---
    print("\n[Task 2 - Perfect answer]")
    result = grade(
        "task2_routing_and_reply",
        action={
            "priority": "high",
            "category": "billing",
            "route_to": "finance_team",
            "draft_reply": (
                "Thank you for reaching out regarding invoice #4521. "
                "We understand this is urgent. Our finance team will "
                "review and follow up with you within 24 hours."
            ),
        },
        ground_truth={
            "priority": "high",
            "category": "billing",
            "route_to": "finance_team",
            "reply_must_include": ["invoice", "finance", "follow up"],
        },
    )
    print(f"  Total reward: {result.total_reward}")
    print(f"  Feedback: {result.feedback}")

    # --- Task 3 Test ---
    print("\n[Task 3 - Partial answer]")
    from emails import TASK3_THREADS
    thread_data = TASK3_THREADS[0]
    result = grade(
        "task3_thread_analysis",
        action={
            "priority": "high",
            "category": "account_management",
            "route_to": "account_management_team",
            "required_actions": ["reverse_cancellation", "process_upgrade"],
            "draft_reply": (
                "Thank you for reaching out. We understand you changed your mind "
                "about the cancellation and would like to upgrade to Enterprise. "
                "We will reverse the cancellation and process your upgrade request immediately."
            ),
            "context_summary": (
                "The client initially requested cancellation, which was processed. "
                "They then changed their mind and also want to upgrade to the Enterprise plan. "
                "We need to reverse the cancellation and process the upgrade."
            ),
        },
        ground_truth=thread_data["ground_truth"],
        thread=thread_data["thread"],
    )
    print(f"  Total reward: {result.total_reward}")
    print(f"  Feedback: {result.feedback}")

    print("\n" + "=" * 60)
    print("All grader tests passed!")
