"""
Baseline Heuristic Agent for the Email Triage Environment.

This agent uses simple keyword-matching rules to classify, prioritize,
and draft responses for emails. It serves as a reproducible baseline
that any learning agent should be able to beat.
"""

from __future__ import annotations

import re
from typing import Dict, List

from models import TriageAction, TriageObservation


# ─── Keyword Rules ────────────────────────────────────────────────────────────

SPAM_SIGNALS = [
    "won", "winner", "prize", "claim", "congratulations", "click here",
    "act now", "limited time", "bank details", "unsubscribe",
    "newsletter", "suspicious-domain", "verify your identity",
    "$1,000,000", "!!!",
]

URGENT_BUG_SIGNALS = [
    "production", "down", "error", "500", "crash", "outage", "bug",
    "broken", "failing", "403", "401", "deleted", "vanished", "gone",
    "data loss",
]

COMPLAINT_SIGNALS = [
    "frustrated", "disappointed", "unacceptable", "terrible",
    "worst", "furious", "angry", "disgusted", "competitor",
    "cancel", "refund", "lawyer", "third time", "repeatedly",
]

BILLING_SIGNALS = [
    "pricing", "plan", "upgrade", "billing", "invoice", "payment",
    "subscription", "tier", "cost", "charge",
]

FEATURE_SIGNALS = [
    "feature", "request", "would be great", "suggestion", "roadmap",
    "dark mode", "add support", "wish list",
]

PARTNERSHIP_SIGNALS = [
    "partnership", "collaboration", "integration", "synergy",
    "joint venture", "series", "investment", "valuation",
    "investor", "vc", "funding",
]

INTERNAL_SIGNALS = [
    "team", "sprint", "roadmap", "hr", "expense", "compliance",
    "confidential", "war room", "internal", "mandatory",
    "@ourcompany.com",
]


def classify_email(subject: str, body: str, sender: str) -> tuple[str, str]:
    """
    Classify email category and priority using keyword matching.

    Returns: (category, priority)
    """
    text = (subject + " " + body + " " + sender).lower()

    # Check spam first
    spam_hits = sum(1 for s in SPAM_SIGNALS if s.lower() in text)
    if spam_hits >= 2:
        return "spam", "low"

    # Check urgent bugs
    bug_hits = sum(1 for s in URGENT_BUG_SIGNALS if s.lower() in text)
    complaint_hits = sum(1 for s in COMPLAINT_SIGNALS if s.lower() in text)

    if bug_hits >= 2:
        priority = "critical" if any(w in text for w in ["production", "down", "deleted", "data loss"]) else "high"
        return "urgent_bug", priority

    if complaint_hits >= 2:
        priority = "high" if any(w in text for w in ["cancel", "refund", "lawyer", "competitor"]) else "medium"
        return "complaint", priority

    # Internal
    if "@ourcompany.com" in sender.lower() or sum(1 for s in INTERNAL_SIGNALS if s.lower() in text) >= 2:
        priority = "critical" if "confidential" in text or "war room" in text else "high" if "roadmap" in text or "sprint" in text else "medium"
        return "internal", priority

    # Partnership
    partnership_hits = sum(1 for s in PARTNERSHIP_SIGNALS if s.lower() in text)
    if partnership_hits >= 2:
        return "partnership", "high"

    # Billing
    billing_hits = sum(1 for s in BILLING_SIGNALS if s.lower() in text)
    if billing_hits >= 2:
        return "billing", "medium"

    # Feature request
    feature_hits = sum(1 for s in FEATURE_SIGNALS if s.lower() in text)
    if feature_hits >= 1:
        return "feature_request", "low"

    # Default
    return "general_inquiry", "medium"


def should_respond(category: str) -> bool:
    """Determine if an email needs a response."""
    return category != "spam"


def pick_tone(category: str, priority: str) -> str:
    """Select appropriate response tone."""
    if category in ("complaint", "urgent_bug") and priority in ("critical", "high"):
        return "apologetic"
    if category in ("partnership", "internal"):
        return "formal"
    if category in ("feature_request", "general_inquiry"):
        return "friendly"
    return "neutral"


def should_escalate(category: str, priority: str) -> bool:
    """Determine if escalation is needed."""
    return priority == "critical" and category in ("urgent_bug", "complaint", "general_inquiry", "internal")


def draft_response(category: str, priority: str, subject: str) -> str:
    """Generate a basic template response."""
    templates = {
        "urgent_bug": (
            "Thank you for reporting this issue. We understand the urgency and have escalated this to our engineering team. "
            "We are actively investigating and will provide an update within the next 2 hours. "
            "We apologize for any disruption this has caused."
        ),
        "complaint": (
            "Thank you for bringing this to our attention. We sincerely apologize for the experience you've had. "
            "Your feedback is important to us and we are looking into this matter with high priority. "
            "A member of our team will follow up with you personally within 24 hours to discuss a resolution."
        ),
        "billing": (
            "Thank you for your inquiry about our pricing plans. We'd be happy to help you understand the options available. "
            "Our team can walk you through the differences and help find the best fit for your needs. "
            "Would you like to schedule a quick call to discuss further?"
        ),
        "feature_request": (
            "Thank you for the great suggestion! We really appreciate you taking the time to share your feedback. "
            "We've added this to our product roadmap for review. While I can't promise a specific timeline, "
            "your input helps us prioritize what matters most to our users."
        ),
        "partnership": (
            "Thank you for reaching out. We're very interested in exploring this opportunity further. "
            "Your proposal aligns well with our strategic direction. I'd like to suggest we schedule a meeting "
            "next week to discuss the details. Please let me know your availability."
        ),
        "internal": (
            "Thank you for the update. Acknowledged and noted. "
            "I will review the details and follow up with any questions or blockers."
        ),
        "general_inquiry": (
            "Thank you for reaching out. We appreciate your question and are happy to help. "
            "Please find the relevant information below, and don't hesitate to ask if you need further assistance."
        ),
        "spam": "",
    }
    return templates.get(category, templates["general_inquiry"])


def process_email(email_id: str, subject: str, body: str, sender: str) -> TriageAction:
    """
    Process a single email using the heuristic baseline agent.

    Returns a TriageAction with classification, priority, and response.
    """
    category, priority = classify_email(subject, body, sender)
    respond = should_respond(category)
    tone = pick_tone(category, priority)
    escalate = should_escalate(category, priority)
    response = draft_response(category, priority, subject) if respond else ""

    return TriageAction(
        email_id=email_id,
        category=category,
        priority=priority,
        response_draft=response,
        response_tone=tone,
        should_respond=respond,
        escalate=escalate,
        tags=[category, priority],
    )


def run_baseline(env) -> Dict:
    """
    Run the baseline agent on all tasks and return scores.

    Args:
        env: EmailTriageEnvironment instance

    Returns:
        Dict with scores for each task
    """
    results = {}

    for task_id in ["easy_triage", "medium_triage", "hard_triage"]:
        # Reset environment
        obs = env.reset(task_id=task_id)

        # Process each email
        for email in obs.emails:
            action = process_email(
                email_id=email.id,
                subject=email.subject,
                body=email.body,
                sender=email.sender,
            )
            obs = env.step(action)

        # Get grader result
        grader_result = env.get_grader_result()
        task_score = grader_result.get("task_score", grader_result["score"])
        results[task_id] = {
            "task_score": task_score,
            "score": task_score,
            "difficulty": task_id.replace("_triage", ""),
            "emails_processed": grader_result["emails_graded"],
            "emails_expected": grader_result["emails_expected"],
            "per_email_scores": [
                {"email_id": e["email_id"], "score": e.get("total_reward", 0.01)}
                for e in grader_result.get("per_email", [])
            ],
        }

    return {
        "baseline": "heuristic_keyword_agent",
        "results": results,
        "summary": {
            "easy": results["easy_triage"]["score"],
            "medium": results["medium_triage"]["score"],
            "hard": results["hard_triage"]["score"],
        },
    }


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from server.environment import EmailTriageEnvironment

    env = EmailTriageEnvironment()
    results = run_baseline(env)

    print(json.dumps(results, indent=2))
    print("\n=== Baseline Summary ===")
    for task, score in results["summary"].items():
        print(f"  {task:8s}: {score:.4f}")
