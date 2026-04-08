---
title: Email Triage
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Email Triage Environment

An **OpenEnv-compliant** environment where AI agents learn to triage emails.

The agent reads emails one at a time (or full email threads) and must:
- Classify the **priority** (critical / high / medium / low)
- Identify the **category** (billing, technical, HR, etc.)
- **Route** the email to the correct team
- **Draft** a professional reply

Three tasks range from simple classification (easy) to complex multi-email
thread analysis with conflicting signals (hard).

---

## Why This Environment?

Email triage is something every company does manually today. Training agents
to do this automatically has immediate real-world value — reducing response
times, routing errors, and workload on human support teams.

This environment models the full complexity of real inbox management:
- Ambiguous urgency signals
- Multi-department routing decisions
- Professional communication drafting
- Context tracking across email threads

---

## Tasks

### Task 1 — Priority Classification `(Easy)`

**Goal:** Read each email and classify its priority and category.

**Input:** A single email (sender, subject, body, timestamp)

**Output:**
```json
{
  "priority": "critical",
  "category": "technical_incident"
}
```

**Scoring:**
- Priority correct → +0.50
- Category correct → +0.50
- Partial credit for priority (one level off = 0.25)

**Episodes:** 5 emails, 1 action per email

---

### Task 2 — Routing + Reply `(Medium)`

**Goal:** Classify the email, route it to the right team, and draft a reply.

**Input:** A single email with routing options

**Output:**
```json
{
  "priority": "high",
  "category": "billing",
  "route_to": "finance_team",
  "draft_reply": "Thank you for reaching out about invoice #4521. Our finance team will follow up within 24 hours."
}
```

**Scoring:**
- Priority → 25%
- Category → 15%
- Routing  → 30%
- Reply quality → 30% (length + professionalism + keyword coverage)

**Episodes:** 5 emails, 1 action per email

---

### Task 3 — Multi-Email Thread Analysis `(Hard)`

**Goal:** Read a full email thread (3–5 emails, possibly contradicting each
other), identify all required actions, route correctly, and reply addressing
every open point.

**Input:** Full email thread with timestamps

**Output:**
```json
{
  "priority": "high",
  "category": "account_management",
  "route_to": "account_management_team",
  "required_actions": ["reverse_cancellation", "process_upgrade"],
  "draft_reply": "We understand you changed your mind about cancellation and wish to upgrade to Enterprise. We will reverse the cancellation and process your upgrade immediately.",
  "context_summary": "Client cancelled on Jan 12, support confirmed cancellation. Client reversed decision on Jan 15 and also wants Enterprise upgrade."
}
```

**Scoring:**
- Priority  → 15%
- Category  → 10%
- Routing   → 20%
- Reply     → 25%
- Actions   → 20%
- Context   → 10%

**Episodes:** 3 threads, 1 action per thread

---

## Action & Observation Spaces

### Valid Priorities
```
critical | high | medium | low
```

### Valid Categories
```
technical_incident | spam | hr_admin | meeting_scheduling | security |
billing | technical_support | business_development | hr_complaint |
press_media | account_management | billing_escalation | recruitment | other
```

### Valid Routes
```
it_support | finance_team | hr_team | sales_team | pr_legal_team |
account_management_team | hr_manager | no_action_needed | other
```

---

## Reward Function

Rewards are **partial** — the agent earns credit for each correct component
even if other components are wrong.

```
Task 1:  priority(0.50) + category(0.50)
Task 2:  priority(0.25) + category(0.15) + routing(0.30) + reply(0.30)
Task 3:  priority(0.15) + category(0.10) + routing(0.20) +
         reply(0.25) + actions(0.20) + context(0.10)
```

Priority scoring uses **distance-based partial credit**:
- Exact match → 1.0
- One level off (e.g. high vs critical) → 0.5
- Two levels off → 0.25
- Completely wrong → 0.0

Reply scoring checks:
- Minimum length (20+ words) → 0.2
- Professional tone → 0.2
- Required keyword coverage → 0.6

---

## Setup & Usage

### Option 1 — Run locally with Python

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/your-username/email-triage-env
cd email-triage-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python server.py

# 4. Run the inference script (in a new terminal)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_token_here"
python inference.py
```

### Option 2 — Run with Docker

```bash
# 1. Build the image
docker build -t email-triage-env .

# 2. Run the container
docker run -p 7860:7860 email-triage-env

# 3. Run inference (in a new terminal)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_token_here"
export SERVER_URL="http://localhost:7860"
python inference.py
```

### Option 3 — Use the HuggingFace Space

The environment is deployed at:
```
https://your-username-email-triage-env.hf.space
```

Call the API directly:
```bash
# Reset
curl -X POST https://your-username-email-triage-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "task1_priority_classification"}'

# Step
curl -X POST https://your-username-email-triage-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "task1_priority_classification",
    "action": {"priority": "critical", "category": "technical_incident"}
  }'

# State
curl https://your-username-email-triage-env.hf.space/state?task_name=task1_priority_classification
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/`      | Environment info |
| GET    | `/health` | Health check |
| GET    | `/tasks`  | List all tasks |
| POST   | `/reset`  | Start new episode |
| POST   | `/step`   | Submit action |
| GET    | `/state`  | Current state |
| GET    | `/docs`   | Interactive API docs (Swagger) |

---

## Baseline Scores

Scores achieved by `Qwen/Qwen2.5-72B-Instruct` on the baseline inference script:

| Task | Difficulty | Score |
|------|-----------|-------|
| task1_priority_classification | Easy   | ~0.85 |
| task2_routing_and_reply       | Medium | ~0.70 |
| task3_thread_analysis         | Hard   | ~0.55 |
| **Overall Average**           |        | **~0.70** |

---

## Project Structure

```
email-triage-env/
├── emails.py         # Email dataset for all 3 tasks
├── tasks.py          # Task definitions + Pydantic models
├── graders.py        # Scoring logic (0.0 - 1.0)
├── environment.py    # Core reset/step/state implementation
├── server.py         # FastAPI server exposing HTTP endpoints
├── inference.py      # LLM agent baseline script
├── openenv.yaml      # OpenEnv metadata
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
└── README.md         # This file
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | Yes | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME`   | Yes | `Qwen/Qwen2.5-72B-Instruct` | Model to use |
| `HF_TOKEN`     | Yes | — | HuggingFace API key |
| `SERVER_URL`   | No  | `http://localhost:7860` | Environment server URL |
| `PORT`         | No  | `7860` | Server port |

---

## License

MIT
