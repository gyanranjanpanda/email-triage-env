#!/usr/bin/env python3
"""
Inference script for the Email Triage Environment.

Uses the OpenAI API client to run an LLM-based agent against the environment.
Reads API credentials from environment variables:
  - API_BASE_URL: The API endpoint for the LLM (default: https://router.huggingface.co/v1)
  - MODEL_NAME: The model identifier for inference
  - HF_TOKEN / OPENAI_API_KEY: Your API key

Usage:
  # Direct mode (default):
  python inference.py

  # Server mode:
  python inference.py --server http://localhost:7860
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


SYSTEM_PROMPT = """You are an expert email triage agent. For each email, you must classify it and decide how to handle it.

You MUST respond with valid JSON only, no other text. Use this exact schema:
{
  "category": "<one of: urgent_bug, feature_request, billing, general_inquiry, spam, complaint, partnership, internal>",
  "priority": "<one of: critical, high, medium, low>",
  "should_respond": <true or false>,
  "response_tone": "<one of: formal, friendly, apologetic, neutral>",
  "escalate": <true or false>,
  "response_draft": "<your draft response or empty string if should_respond is false>"
}

Classification guidelines:
- spam: Unsolicited, scam-like, prize notifications, phishing
- urgent_bug: Production issues, system outages, critical errors
- complaint: Customer frustration, service dissatisfaction, cancellation threats
- billing: Pricing questions, invoices, payment issues, plan upgrades
- feature_request: Product suggestions, enhancement ideas, wishlist items
- partnership: Business collaborations, integration proposals, investment
- internal: Company-internal communications, HR, compliance
- general_inquiry: Everything else

Priority guidelines:
- critical: Production down, data loss, security breach, legal threat
- high: Major customer impact, executive escalation, partnership opportunity
- medium: Standard business inquiries, moderate complaints
- low: Feature requests, spam, general questions

Escalation: Set to true for critical priority items or legal/PR-sensitive situations.
Response tone: Use apologetic for complaints/bugs, formal for partnerships/legal, friendly for features/inquiries."""


def llm_process_email(email_id: str, subject: str, body: str, sender: str) -> dict:
    """Use the OpenAI-compatible LLM to triage a single email."""
    user_message = f"""Triage this email:

From: {sender}
Subject: {subject}
Body: {body}

Respond with JSON only."""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        raw_response = completion.choices[0].message.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in raw_response:
            raw_response = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            raw_response = raw_response.split("```")[1].split("```")[0].strip()

        parsed = json.loads(raw_response)

        return {
            "email_id": email_id,
            "category": parsed.get("category", "general_inquiry"),
            "priority": parsed.get("priority", "medium"),
            "should_respond": parsed.get("should_respond", True),
            "response_tone": parsed.get("response_tone", "neutral"),
            "escalate": parsed.get("escalate", False),
            "response_draft": parsed.get("response_draft", ""),
            "tags": [parsed.get("category", ""), parsed.get("priority", "")],
        }
    except Exception as exc:
        # Fallback to safe defaults on LLM failure
        print(f"  [WARN] LLM call failed for {email_id}: {exc}. Using fallback.")
        return {
            "email_id": email_id,
            "category": "general_inquiry",
            "priority": "medium",
            "should_respond": True,
            "response_tone": "neutral",
            "escalate": False,
            "response_draft": "Thank you for your email. We will review and get back to you shortly.",
            "tags": ["fallback"],
        }


def run_direct():
    """Run LLM-based agent directly against the environment (no server)."""
    from server.environment import EmailTriageEnvironment
    from models import TriageAction

    env = EmailTriageEnvironment()
    results = {}

    for task_id in ["easy_triage", "medium_triage", "hard_triage"]:
        print(f"[START] task={task_id}", flush=True)
        obs = env.reset(task_id=task_id)
        step_num = 0

        for email in obs.emails:
            action_dict = llm_process_email(
                email_id=email.id,
                subject=email.subject,
                body=email.body,
                sender=email.sender,
            )
            action = TriageAction(**action_dict)
            obs = env.step(action)
            step_num += 1
            print(f"[STEP] step={step_num} reward={obs.reward}", flush=True)

        grader_result = env.get_grader_result()
        task_score = grader_result["score"]
        print(f"[END] task={task_id} score={task_score} steps={step_num}", flush=True)

        results[task_id] = {
            "score": task_score,
            "difficulty": task_id.replace("_triage", ""),
            "emails_processed": grader_result["emails_graded"],
            "emails_expected": grader_result["emails_expected"],
            "per_email_scores": [
                {"email_id": e["email_id"], "score": e.get("total_reward", 0.0)}
                for e in grader_result.get("per_email", [])
            ],
        }

    return {
        "baseline": "llm_openai_agent",
        "results": results,
        "summary": {
            "easy": results["easy_triage"]["score"],
            "medium": results["medium_triage"]["score"],
            "hard": results["hard_triage"]["score"],
        },
    }


def run_against_server(base_url: str):
    """Run LLM-based agent against a running HTTP server."""
    import httpx

    http_client = httpx.Client(base_url=base_url, timeout=30.0)
    results = {}

    for task_id in ["easy_triage", "medium_triage", "hard_triage"]:
        print(f"[START] task={task_id}", flush=True)
        resp = http_client.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
        obs = resp.json()
        step_num = 0

        for email in obs["emails"]:
            action_dict = llm_process_email(
                email_id=email["id"],
                subject=email["subject"],
                body=email["body"],
                sender=email["sender"],
            )
            step_resp = http_client.post("/step", json=action_dict)
            step_resp.raise_for_status()
            step_data = step_resp.json()
            step_num += 1
            print(f"[STEP] step={step_num} reward={step_data['reward']}", flush=True)

        grader_resp = http_client.post("/grader")
        grader_resp.raise_for_status()
        grader_result = grader_resp.json()
        task_score = grader_result["score"]
        print(f"[END] task={task_id} score={task_score} steps={step_num}", flush=True)

        results[task_id] = {
            "score": task_score,
            "difficulty": task_id.replace("_triage", ""),
            "emails_processed": grader_result["emails_graded"],
            "emails_expected": grader_result["emails_expected"],
        }

    return {
        "baseline": "llm_openai_agent",
        "results": results,
        "summary": {
            "easy": results["easy_triage"]["score"],
            "medium": results["medium_triage"]["score"],
            "hard": results["hard_triage"]["score"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run Email Triage LLM inference")
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="Base URL of running server (e.g., http://localhost:7860). If not set, runs directly.",
    )
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("  Email Triage Environment — LLM Inference", flush=True)
    print("=" * 60, flush=True)
    print(f"  API_BASE_URL: {API_BASE_URL}", flush=True)
    print(f"  MODEL_NAME:   {MODEL_NAME}", flush=True)
    print(f"  API_KEY:      {'***' + API_KEY[-4:] if len(API_KEY) > 4 else '(not set)'}", flush=True)

    if args.server:
        print(f"\n  Mode: HTTP (server={args.server})", flush=True)
        results = run_against_server(args.server)
    else:
        print("\n  Mode: Direct (no server)", flush=True)
        results = run_direct()

    print("\n" + json.dumps(results, indent=2), flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  Baseline Scores", flush=True)
    print("=" * 60, flush=True)
    for task, score in results["summary"].items():
        bar = "█" * int(score * 40) + "░" * (40 - int(score * 40))
        print(f"  {task:8s} │ {bar} │ {score:.4f}", flush=True)
    print("=" * 60, flush=True)

    avg = sum(results["summary"].values()) / len(results["summary"])
    print(f"  Average: {avg:.4f}", flush=True)
    print(flush=True)

    for task, score in results["summary"].items():
        assert 0.0 <= score <= 1.0, f"Score out of range for {task}: {score}"

    print("✓ All scores in valid range [0.0, 1.0]", flush=True)
    print("✓ Inference complete", flush=True)


if __name__ == "__main__":
    main()
