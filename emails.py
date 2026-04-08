"""
emails.py
---------
Sample email dataset for the Email Triage Environment.

Contains emails for all 3 tasks:
  - Task 1: Priority Classification (Easy)
  - Task 2: Priority + Routing + Reply (Medium)
  - Task 3: Multi-email Thread (Hard)
"""

# ============================================================
# TASK 1 EMAILS — Priority Classification (Easy)
# Agent must identify: priority + category
# ============================================================

TASK1_EMAILS = [
    {
        "email_id": "t1_001",
        "sender": "boss@company.com",
        "subject": "URGENT - Production server is down",
        "body": (
            "Our production server crashed 10 minutes ago. "
            "Customers cannot log in. We are losing revenue every minute. "
            "All hands on deck. Fix this immediately."
        ),
        "timestamp": "2024-01-15T09:00:00",
        # Ground truth — what the agent SHOULD answer
        "ground_truth": {
            "priority": "critical",
            "category": "technical_incident",
        },
    },
    {
        "email_id": "t1_002",
        "sender": "newsletter@randomsite.com",
        "subject": "Top 10 productivity tips you need to know!",
        "body": (
            "Hi there! Check out our latest blog post on productivity. "
            "Click here to read more. Unsubscribe at any time."
        ),
        "timestamp": "2024-01-15T09:05:00",
        "ground_truth": {
            "priority": "low",
            "category": "spam",
        },
    },
    {
        "email_id": "t1_003",
        "sender": "hr@company.com",
        "subject": "Reminder: Submit your timesheet by Friday",
        "body": (
            "Hi team, just a friendly reminder to submit your timesheets "
            "by end of day Friday. Please log in to the HR portal to complete this."
        ),
        "timestamp": "2024-01-15T09:10:00",
        "ground_truth": {
            "priority": "medium",
            "category": "hr_admin",
        },
    },
    {
        "email_id": "t1_004",
        "sender": "ceo@company.com",
        "subject": "Board meeting moved to tomorrow 9AM",
        "body": (
            "Important: The board meeting has been rescheduled to tomorrow at 9AM. "
            "All department heads must attend. Please confirm your availability immediately."
        ),
        "timestamp": "2024-01-15T09:15:00",
        "ground_truth": {
            "priority": "high",
            "category": "meeting_scheduling",
        },
    },
    {
        "email_id": "t1_005",
        "sender": "security@company.com",
        "subject": "ALERT: Suspicious login detected on your account",
        "body": (
            "We detected a suspicious login attempt on your account from an unknown location. "
            "If this was not you, please reset your password immediately and contact IT security."
        ),
        "timestamp": "2024-01-15T09:20:00",
        "ground_truth": {
            "priority": "critical",
            "category": "security",
        },
    },
]


# ============================================================
# TASK 2 EMAILS — Priority + Routing + Reply Draft (Medium)
# Agent must identify: priority + category + route_to + draft_reply
# ============================================================

TASK2_EMAILS = [
    {
        "email_id": "t2_001",
        "sender": "client@bigcorp.com",
        "subject": "Invoice #4521 overdue - need clarification",
        "body": (
            "Hi, we have not received payment confirmation for invoice #4521 "
            "sent 3 weeks ago. The amount is $12,500. Please advise on the status "
            "as this is now overdue and affecting our accounting."
        ),
        "timestamp": "2024-01-15T10:00:00",
        "ground_truth": {
            "priority": "high",
            "category": "billing",
            "route_to": "finance_team",
            # Key topics the reply MUST address
            "reply_must_include": ["invoice", "finance", "follow up"],
        },
    },
    {
        "email_id": "t2_002",
        "sender": "newuser@gmail.com",
        "subject": "Cannot reset my password - locked out",
        "body": (
            "Hello, I have been trying to reset my password for the past 2 hours "
            "but the reset email never arrives. I have checked my spam folder. "
            "I cannot access my account and have an important deadline today."
        ),
        "timestamp": "2024-01-15T10:05:00",
        "ground_truth": {
            "priority": "high",
            "category": "technical_support",
            "route_to": "it_support",
            "reply_must_include": ["password", "support", "help"],
        },
    },
    {
        "email_id": "t2_003",
        "sender": "partner@vendorcorp.com",
        "subject": "New partnership proposal - Q2 collaboration",
        "body": (
            "Dear team, we would like to propose a strategic partnership for Q2. "
            "We believe our services complement yours well. "
            "Could we schedule a call next week to discuss the details?"
        ),
        "timestamp": "2024-01-15T10:10:00",
        "ground_truth": {
            "priority": "medium",
            "category": "business_development",
            "route_to": "sales_team",
            "reply_must_include": ["partnership", "schedule", "call"],
        },
    },
    {
        "email_id": "t2_004",
        "sender": "employee@company.com",
        "subject": "Harassment complaint - urgent and confidential",
        "body": (
            "I need to report a serious harassment incident that occurred yesterday. "
            "I feel unsafe and uncomfortable at work. "
            "I would like to speak with someone from HR as soon as possible. "
            "Please treat this as confidential."
        ),
        "timestamp": "2024-01-15T10:15:00",
        "ground_truth": {
            "priority": "critical",
            "category": "hr_complaint",
            "route_to": "hr_team",
            "reply_must_include": ["confidential", "hr", "priority"],
        },
    },
    {
        "email_id": "t2_005",
        "sender": "press@newsagency.com",
        "subject": "Request for comment on recent data breach reports",
        "body": (
            "Hi, I am a journalist covering cybersecurity. "
            "There are reports circulating that your company suffered a data breach last week. "
            "Could you provide an official comment for our article publishing tomorrow?"
        ),
        "timestamp": "2024-01-15T10:20:00",
        "ground_truth": {
            "priority": "critical",
            "category": "press_media",
            "route_to": "pr_legal_team",
            "reply_must_include": ["comment", "team", "respond"],
        },
    },
]


# ============================================================
# TASK 3 EMAILS — Multi-Email Thread with Conflicting Signals (Hard)
# Agent must read a full thread and make a nuanced decision
# ============================================================

TASK3_THREADS = [
    {
        "thread_id": "t3_001",
        "subject": "Subscription cancellation request",
        # List of emails in the thread, in chronological order
        "thread": [
            {
                "from": "client@acme.com",
                "timestamp": "2024-01-12T14:00:00",
                "body": "Please cancel my subscription effective immediately. I am switching to a competitor.",
            },
            {
                "from": "support@company.com",
                "timestamp": "2024-01-13T09:00:00",
                "body": "Hi, your cancellation has been processed. Your access will end on Jan 31. Sorry to see you go.",
            },
            {
                "from": "client@acme.com",
                "timestamp": "2024-01-15T08:00:00",
                "body": (
                    "Wait — I changed my mind. Please do NOT cancel my subscription. "
                    "The competitor did not work out. Also, I actually want to UPGRADE "
                    "to the Enterprise plan. Can you help with both?"
                ),
            },
        ],
        "ground_truth": {
            "priority": "high",
            "category": "account_management",
            "route_to": "account_management_team",
            "required_actions": ["reverse_cancellation", "process_upgrade"],
            "reply_must_include": ["cancellation", "upgrade", "enterprise"],
            "context_understood": True,  # Agent must show it read all 3 emails
        },
    },
    {
        "thread_id": "t3_002",
        "subject": "Refund request for order #8821",
        "thread": [
            {
                "from": "customer@shop.com",
                "timestamp": "2024-01-10T10:00:00",
                "body": "I want a refund for order #8821. The product arrived damaged.",
            },
            {
                "from": "support@company.com",
                "timestamp": "2024-01-11T09:00:00",
                "body": "We are sorry to hear that. We have initiated a refund. It will appear in 5-7 business days.",
            },
            {
                "from": "customer@shop.com",
                "timestamp": "2024-01-14T11:00:00",
                "body": "It has been 5 days and no refund yet. Also, I want to reorder the same item — is it back in stock?",
            },
            {
                "from": "support@company.com",
                "timestamp": "2024-01-14T15:00:00",
                "body": "The refund is still processing. We will escalate this to billing.",
            },
            {
                "from": "customer@shop.com",
                "timestamp": "2024-01-15T09:00:00",
                "body": (
                    "This is unacceptable. It has now been 7 business days. "
                    "No refund, no update. I am going to dispute this with my credit card company "
                    "if I do not hear back today. I also still want to know about the reorder."
                ),
            },
        ],
        "ground_truth": {
            "priority": "critical",
            "category": "billing_escalation",
            "route_to": "finance_team",
            "required_actions": ["escalate_refund", "check_stock"],
            "reply_must_include": ["refund", "escalate", "reorder"],
            "context_understood": True,
        },
    },
    {
        "thread_id": "t3_003",
        "subject": "Job offer negotiation",
        "thread": [
            {
                "from": "hr@company.com",
                "timestamp": "2024-01-08T10:00:00",
                "body": "We are pleased to offer you the Senior Engineer position at $120,000/year. Please confirm by Jan 15.",
            },
            {
                "from": "candidate@email.com",
                "timestamp": "2024-01-09T14:00:00",
                "body": "Thank you for the offer. I was hoping for $135,000 given my 8 years of experience. Is there flexibility?",
            },
            {
                "from": "hr@company.com",
                "timestamp": "2024-01-10T09:00:00",
                "body": "We can stretch to $128,000 as our final offer. We hope you will accept.",
            },
            {
                "from": "candidate@email.com",
                "timestamp": "2024-01-15T08:00:00",
                "body": (
                    "I appreciate the offer but I have received a competing offer at $132,000. "
                    "I genuinely prefer your company. Is there any possibility of matching this, "
                    "or perhaps adding additional vacation days or a signing bonus instead?"
                ),
            },
        ],
        "ground_truth": {
            "priority": "high",
            "category": "recruitment",
            "route_to": "hr_manager",
            "required_actions": ["escalate_to_hiring_manager", "consider_counter_offer"],
            "reply_must_include": ["offer", "manager", "respond"],
            "context_understood": True,
        },
    },
]


# ============================================================
# HELPER — Valid values the agent can use
# ============================================================

VALID_PRIORITIES = ["critical", "high", "medium", "low"]

VALID_CATEGORIES = [
    "technical_incident",
    "spam",
    "hr_admin",
    "meeting_scheduling",
    "security",
    "billing",
    "technical_support",
    "business_development",
    "hr_complaint",
    "press_media",
    "account_management",
    "billing_escalation",
    "recruitment",
    "other",
]

VALID_ROUTES = [
    "it_support",
    "finance_team",
    "hr_team",
    "sales_team",
    "pr_legal_team",
    "account_management_team",
    "hr_manager",
    "no_action_needed",
    "other",
]
