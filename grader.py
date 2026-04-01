"""
Grading logic for the Email Triage Environment.

Evaluates agent actions against ground truth across multiple dimensions:
- Classification accuracy (category + priority)
- Response appropriateness (tone, content quality)
- Decision making (respond vs. not respond, escalation)
"""

from __future__ import annotations

from typing import Dict, List

from models import TriageAction


def grade_classification(action: TriageAction, truth: Dict) -> float:
    """
    Grade the classification of an email (category + priority).
    Returns score in [0.0, 1.0].
    """
    score = 0.0

    # Category match (0.6 weight)
    if action.category == truth["category"]:
        score += 0.6
    else:
        # Partial credit for close categories
        partial_map = {
            ("urgent_bug", "complaint"): 0.2,
            ("complaint", "urgent_bug"): 0.2,
            ("general_inquiry", "feature_request"): 0.15,
            ("feature_request", "general_inquiry"): 0.15,
            ("billing", "general_inquiry"): 0.1,
            ("general_inquiry", "billing"): 0.1,
            ("partnership", "general_inquiry"): 0.1,
        }
        key = (action.category, truth["category"])
        score += partial_map.get(key, 0.0)

    # Priority match (0.4 weight)
    priority_order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    agent_p = priority_order.get(action.priority, -1)
    true_p = priority_order.get(truth["priority"], -1)

    if agent_p == true_p:
        score += 0.4
    else:
        # Partial credit based on distance
        diff = abs(agent_p - true_p)
        if diff == 1:
            score += 0.2
        elif diff == 2:
            score += 0.05

    return min(score, 1.0)


def grade_response_decision(action: TriageAction, truth: Dict) -> float:
    """
    Grade whether the agent correctly decided to respond or not.
    Returns score in [0.0, 1.0].
    """
    if action.should_respond == truth["requires_response"]:
        return 1.0

    # Responding to spam is worse than not responding to something that needs it
    if action.should_respond and not truth["requires_response"]:
        if truth["category"] == "spam":
            return 0.1  # Responding to spam is bad
        return 0.5  # Responding when not needed is okay-ish

    # Not responding when needed
    if truth["priority"] in ("critical", "high"):
        return 0.0  # Ignoring critical/high priority is very bad
    return 0.3  # Ignoring medium/low is less bad


def grade_response_tone(action: TriageAction, truth: Dict) -> float:
    """
    Grade whether the response tone is appropriate.
    Returns score in [0.0, 1.0].
    """
    if not truth["requires_response"]:
        return 1.0  # Tone doesn't matter if no response needed

    if not action.should_respond or not action.response_draft:
        return 0.0  # No response to grade

    if action.response_tone == truth["tone"]:
        return 1.0

    # Partial credit for acceptable tones
    acceptable = {
        "apologetic": {"formal": 0.5, "friendly": 0.4, "neutral": 0.3},
        "formal": {"neutral": 0.6, "friendly": 0.4, "apologetic": 0.5},
        "friendly": {"neutral": 0.6, "formal": 0.5, "apologetic": 0.4},
        "neutral": {"formal": 0.7, "friendly": 0.6, "apologetic": 0.4},
    }

    return acceptable.get(truth["tone"], {}).get(action.response_tone, 0.2)


def grade_response_content(action: TriageAction, truth: Dict) -> float:
    """
    Grade the quality of the response draft based on key points.
    Returns score in [0.0, 1.0].
    """
    if not truth["requires_response"]:
        if not action.should_respond or not action.response_draft:
            return 1.0
        return 0.7  # Unnecessary response but not terrible

    if not action.response_draft:
        return 0.0

    draft_lower = action.response_draft.lower()
    key_points = truth.get("key_points", [])

    if not key_points:
        # No specific points to check — just verify response exists and is reasonable
        if len(action.response_draft) > 20:
            return 0.8
        return 0.4

    # Check key points coverage
    points_hit = 0
    for point in key_points:
        point_keywords = point.lower().split()
        # Check if the draft addresses this point (keyword matching)
        matches = sum(1 for kw in point_keywords if kw in draft_lower)
        if matches >= len(point_keywords) * 0.4:  # At least 40% keyword match
            points_hit += 1

    coverage = points_hit / len(key_points) if key_points else 0

    # Also check response length (too short = bad, reasonable = good)
    length_score = min(len(action.response_draft) / 100, 1.0) * 0.3

    return min(coverage * 0.7 + length_score, 1.0)


def grade_escalation(action: TriageAction, truth: Dict) -> float:
    """
    Grade whether the agent correctly decided to escalate.
    Returns score in [0.0, 1.0].
    """
    should_escalate = truth["priority"] == "critical" and truth["category"] in (
        "urgent_bug", "complaint", "general_inquiry"
    )
    key_points = truth.get("key_points", [])
    escalation_mentioned = any("escalat" in kp.lower() for kp in key_points)
    should_escalate = should_escalate or escalation_mentioned

    if action.escalate == should_escalate:
        return 1.0

    if action.escalate and not should_escalate:
        return 0.6  # Over-escalating is cautious, not terrible

    # Not escalating when should have
    if truth["priority"] == "critical":
        return 0.1
    return 0.4


def grade_action(action: TriageAction, truth: Dict, difficulty: str = "easy") -> Dict:
    """
    Compute full grading for a single action.

    Returns dict with component scores and total reward.
    """
    classification = grade_classification(action, truth)
    response_decision = grade_response_decision(action, truth)
    tone = grade_response_tone(action, truth)
    content = grade_response_content(action, truth)
    escalation = grade_escalation(action, truth)

    # Weights vary by difficulty
    if difficulty == "easy":
        weights = {
            "classification": 0.50,
            "response_decision": 0.25,
            "tone": 0.10,
            "content": 0.10,
            "escalation": 0.05,
        }
    elif difficulty == "medium":
        weights = {
            "classification": 0.30,
            "response_decision": 0.20,
            "tone": 0.15,
            "content": 0.20,
            "escalation": 0.15,
        }
    else:  # hard
        weights = {
            "classification": 0.20,
            "response_decision": 0.15,
            "tone": 0.20,
            "content": 0.25,
            "escalation": 0.20,
        }

    total = (
        classification * weights["classification"]
        + response_decision * weights["response_decision"]
        + tone * weights["tone"]
        + content * weights["content"]
        + escalation * weights["escalation"]
    )

    return {
        "classification": round(classification, 3),
        "response_decision": round(response_decision, 3),
        "tone": round(tone, 3),
        "content": round(content, 3),
        "escalation": round(escalation, 3),
        "total_reward": round(total, 3),
        "weights": weights,
    }


def grade_episode(
    actions: List[TriageAction],
    truths: List[Dict],
    difficulty: str = "easy",
) -> Dict:
    """
    Grade an entire episode (batch of emails).

    Returns dict with per-email scores and aggregate score.
    """
    if not actions or not truths:
        return {"score": 0.0, "per_email": [], "summary": "No actions to grade"}

    per_email = []
    total_score = 0.0

    # Match actions to truths by email_id
    truth_map = {t["email_id"]: t for t in truths}

    for action in actions:
        truth = truth_map.get(action.email_id)
        if truth is None:
            per_email.append({
                "email_id": action.email_id,
                "error": "No matching email found",
                "score": 0.0,
            })
            continue

        grades = grade_action(action, truth, difficulty)
        per_email.append({
            "email_id": action.email_id,
            **grades,
        })
        total_score += grades["total_reward"]

    # Normalize by number of expected emails (not just actions submitted)
    n_expected = len(truths)
    avg_score = total_score / n_expected if n_expected > 0 else 0.0

    return {
        "score": round(avg_score, 4),
        "per_email": per_email,
        "emails_graded": len(per_email),
        "emails_expected": n_expected,
        "difficulty": difficulty,
    }
