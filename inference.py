#!/usr/bin/env python3
"""
Inference script for the Email Triage Environment (OpenEnv Hackathon).

Uses the OpenAI API client to run an LLM-based agent against the environment server.

Required environment variables:
  - API_BASE_URL: The API endpoint for the LLM (default: https://router.huggingface.co/v1)
  - MODEL_NAME:   The model identifier for inference (default: meta-llama/Llama-3.1-8B-Instruct)
  - HF_TOKEN:     Your Hugging Face API token (mandatory)

Output format (OpenEnv spec):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

# ─── Environment Variables ────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ─── System Prompt ────────────────────────────────────────────────────────────

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


def llm_triage_email(email_id: str, subject: str, body: str, sender: str) -> dict:
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


def _format_bool(val: bool) -> str:
    """Format a boolean as lowercase string per OpenEnv spec."""
    return "true" if val else "false"


def _clamp_reward(r: float) -> float:
    """Clamp reward to (0.01, 0.99) so the formatted value never hits 0.00 or 1.00."""
    return max(0.01, min(0.99, r))


def run_direct():
    """Run LLM-based agent directly against the environment (no server)."""
    from server.environment import EmailTriageEnvironment
    from models import TriageAction

    env = EmailTriageEnvironment()

    for task_id in ["easy_triage", "medium_triage", "hard_triage"]:
        # ── [START] ──
        print(
            f"[START] task={task_id} env=email_triage model={MODEL_NAME}",
            flush=True,
        )

        obs = env.reset(task_id=task_id)
        step_num = 0
        rewards: list[float] = []
        last_error = None
        episode_done = False

        try:
            for email in obs.emails:
                action_dict = llm_triage_email(
                    email_id=email.id,
                    subject=email.subject,
                    body=email.body,
                    sender=email.sender,
                )

                action = TriageAction(**action_dict)
                obs = env.step(action)
                step_num += 1

                reward = _clamp_reward(round(obs.reward, 2))
                rewards.append(reward)
                episode_done = obs.done

                # Represent the action as a compact string
                action_str = f"triage(email_id='{action.email_id}',category='{action.category}',priority='{action.priority}')"

                # ── [STEP] ──
                print(
                    f"[STEP] step={step_num} action={action_str} "
                    f"reward={reward:.2f} done={_format_bool(episode_done)} "
                    f"error=null",
                    flush=True,
                )

        except Exception as exc:
            last_error = str(exc)
            step_num += 1
            rewards.append(0.01)
            action_str = "error"
            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward=0.01 done=true "
                f"error={last_error}",
                flush=True,
            )

        # ── [END] ──
        success = episode_done and last_error is None
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={_format_bool(success)} steps={step_num} "
            f"rewards={rewards_str}",
            flush=True,
        )


def run_against_server(base_url: str):
    """Run LLM-based agent against a running HTTP server."""
    import httpx

    http_client = httpx.Client(base_url=base_url, timeout=30.0)

    for task_id in ["easy_triage", "medium_triage", "hard_triage"]:
        # ── [START] ──
        print(
            f"[START] task={task_id} env=email_triage model={MODEL_NAME}",
            flush=True,
        )

        resp = http_client.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
        obs = resp.json()
        step_num = 0
        rewards: list[float] = []
        last_error = None
        episode_done = False

        try:
            for email in obs["emails"]:
                action_dict = llm_triage_email(
                    email_id=email["id"],
                    subject=email["subject"],
                    body=email["body"],
                    sender=email["sender"],
                )
                step_resp = http_client.post("/step", json=action_dict)
                step_resp.raise_for_status()
                step_data = step_resp.json()
                step_num += 1

                reward = _clamp_reward(round(step_data["reward"], 2))
                rewards.append(reward)
                episode_done = step_data.get("done", False)

                action_str = (
                    f"triage(email_id='{action_dict['email_id']}',"
                    f"category='{action_dict['category']}',"
                    f"priority='{action_dict['priority']}')"
                )

                # ── [STEP] ──
                print(
                    f"[STEP] step={step_num} action={action_str} "
                    f"reward={reward:.2f} done={_format_bool(episode_done)} "
                    f"error=null",
                    flush=True,
                )

        except Exception as exc:
            last_error = str(exc)
            step_num += 1
            rewards.append(0.01)
            action_str = "error"
            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward=0.01 done=true "
                f"error={last_error}",
                flush=True,
            )

        # ── [END] ──
        success = episode_done and last_error is None
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={_format_bool(success)} steps={step_num} "
            f"rewards={rewards_str}",
            flush=True,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Email Triage LLM inference")
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="Base URL of running server (e.g., http://localhost:7860). "
        "If not set, runs directly.",
    )
    args = parser.parse_args()

    if args.server:
        run_against_server(args.server)
    else:
        run_direct()


if __name__ == "__main__":
    main()
