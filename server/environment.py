"""
Email Triage Environment - Core environment logic.

Implements the OpenEnv Environment interface with step(), reset(), and state().
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from models import (
    Email,
    TriageAction,
    TriageObservation,
    TriageState,
)
from email_data import generate_email_batch
from grader import grade_action, grade_episode


def _clamp(val: float) -> float:
    """Clamp a score/reward to (0.01, 0.99) for strict (0, 1) compliance."""
    return max(0.01, min(0.99, val))


# ─── Task definitions ─────────────────────────────────────────────────────────

TASKS = {
    "easy_triage": {
        "id": "easy_triage",
        "name": "Basic Email Triage",
        "description": "Classify and prioritize 5 straightforward emails. Includes obvious spam, clear bug reports, and simple inquiries.",
        "difficulty": "easy",
        "num_emails": 5,
        "max_steps": 5,
        "seed": 42,
    },
    "medium_triage": {
        "id": "medium_triage",
        "name": "Intermediate Email Triage",
        "description": "Process 7 emails including ambiguous cases, complaints requiring empathy, and phishing attempts mixed with legitimate inquiries.",
        "difficulty": "medium",
        "num_emails": 7,
        "max_steps": 7,
        "seed": 123,
    },
    "hard_triage": {
        "id": "hard_triage",
        "name": "Advanced Email Triage",
        "description": "Handle 10 emails including legal notices, press inquiries about data breaches, executive escalations, and high-stakes customer retention scenarios.",
        "difficulty": "hard",
        "num_emails": 10,
        "max_steps": 10,
        "seed": 456,
    },
}


class EmailTriageEnvironment:
    """
    Email Triage & Response Environment.

    An AI agent processes an inbox of emails and must:
    1. Classify each email by category
    2. Assign priority level
    3. Decide whether to respond
    4. Draft appropriate responses with correct tone
    5. Decide whether to escalate

    Implements the OpenEnv spec: step(), reset(), state().
    """

    def __init__(self):
        self._state = TriageState()
        self._emails: List[Email] = []
        self._ground_truths: List[Dict] = []
        self._actions_taken: List[TriageAction] = []
        self._current_task: Optional[Dict] = None
        self._action_grades: List[Dict] = []

    def reset(self, task_id: str = "easy_triage", seed: Optional[int] = None) -> TriageObservation:
        """
        Initialize a new episode.

        Args:
            task_id: One of "easy_triage", "medium_triage", "hard_triage"
            seed: Optional random seed for reproducibility

        Returns:
            Initial observation with the email batch to process.
        """
        task = TASKS.get(task_id, TASKS["easy_triage"])
        self._current_task = task

        effective_seed = seed if seed is not None else task["seed"]

        emails, truths = generate_email_batch(
            difficulty=task["difficulty"],
            batch_size=task["num_emails"],
            seed=effective_seed,
        )

        self._emails = emails
        self._ground_truths = truths
        self._actions_taken = []
        self._action_grades = []

        episode_id = str(uuid.uuid4())
        self._state = TriageState(
            episode_id=episode_id,
            task_id=task_id,
            step_count=0,
            max_steps=task["max_steps"],
            total_reward=0.0,
            emails_processed=0,
            total_emails=len(emails),
            classification_accuracy=0.0,
            response_quality=0.0,
            done=False,
        )

        return TriageObservation(
            emails=emails,
            processed_count=0,
            total_emails=len(emails),
            current_score=0.01,
            feedback=f"Episode started. You have {len(emails)} emails to process. Task: {task['name']}",
            done=False,
            reward=0.01,
            metadata={
                "task_id": task_id,
                "difficulty": task["difficulty"],
                "episode_id": episode_id,
            },
        )

    def step(self, action: TriageAction) -> TriageObservation:
        """
        Process one email triage action.

        Args:
            action: The agent's triage action for one email.

        Returns:
            Observation with feedback, reward, and remaining emails.
        """
        if self._state.done:
            return TriageObservation(
                emails=[],
                processed_count=self._state.emails_processed,
                total_emails=self._state.total_emails,
                current_score=_clamp(self._state.total_reward / max(self._state.emails_processed, 1)),
                feedback="Episode is already complete. Call reset() to start a new episode.",
                done=True,
                reward=0.01,
                metadata={"episode_id": self._state.episode_id},
            )

        # Find the ground truth for this email
        truth = None
        for t in self._ground_truths:
            if t["email_id"] == action.email_id:
                truth = t
                break

        if truth is None:
            return TriageObservation(
                emails=self._get_remaining_emails(),
                processed_count=self._state.emails_processed,
                total_emails=self._state.total_emails,
                current_score=_clamp(self._state.total_reward / max(self._state.emails_processed, 1)),
                feedback=f"Error: Email ID '{action.email_id}' not found in current batch.",
                done=False,
                reward=0.01,
                metadata={"error": "invalid_email_id"},
            )

        # Check for duplicate processing
        already_processed = any(a.email_id == action.email_id for a in self._actions_taken)
        if already_processed:
            return TriageObservation(
                emails=self._get_remaining_emails(),
                processed_count=self._state.emails_processed,
                total_emails=self._state.total_emails,
                current_score=_clamp(self._state.total_reward / max(self._state.emails_processed, 1)),
                feedback=f"Email '{action.email_id}' was already processed. Process a different email.",
                done=False,
                reward=0.01,
                metadata={"error": "duplicate_action"},
            )

        # Grade the action
        difficulty = self._current_task["difficulty"] if self._current_task else "easy"
        grades = grade_action(action, truth, difficulty)

        self._actions_taken.append(action)
        self._action_grades.append(grades)

        # Update state
        self._state.step_count += 1
        self._state.emails_processed += 1
        self._state.total_reward += grades["total_reward"]

        # Compute running averages
        if self._action_grades:
            self._state.classification_accuracy = sum(
                g["classification"] for g in self._action_grades
            ) / len(self._action_grades)
            self._state.response_quality = sum(
                g["content"] for g in self._action_grades
            ) / len(self._action_grades)

        # Check if episode is done
        all_processed = self._state.emails_processed >= self._state.total_emails
        max_steps_reached = self._state.step_count >= self._state.max_steps
        self._state.done = all_processed or max_steps_reached

        # Build feedback
        feedback_parts = [
            f"Email '{action.email_id}' processed.",
            f"Classification: {grades['classification']:.1%}",
            f"Response quality: {grades['content']:.1%}",
            f"Step reward: {grades['total_reward']:.3f}",
        ]
        if self._state.done:
            avg = self._state.total_reward / max(self._state.emails_processed, 1)
            feedback_parts.append(
                f"\nEpisode complete! Average score: {avg:.3f}"
            )

        remaining = self._get_remaining_emails()

        return TriageObservation(
            emails=remaining,
            processed_count=self._state.emails_processed,
            total_emails=self._state.total_emails,
            current_score=_clamp(self._state.total_reward / max(self._state.emails_processed, 1)),
            feedback=" | ".join(feedback_parts),
            done=self._state.done,
            reward=_clamp(grades["total_reward"]),
            metadata={
                "grades": grades,
                "episode_id": self._state.episode_id,
            },
        )

    @property
    def state(self) -> TriageState:
        """Get current episode state."""
        return self._state

    def get_grader_result(self) -> Dict:
        """
        Run the episode grader and return final scores.

        This is called after an episode completes to get the final grade.
        """
        difficulty = self._current_task["difficulty"] if self._current_task else "easy"
        result = grade_episode(self._actions_taken, self._ground_truths, difficulty)
        return result

    def _get_remaining_emails(self) -> List[Email]:
        """Get emails that haven't been processed yet."""
        processed_ids = {a.email_id for a in self._actions_taken}
        return [e for e in self._emails if e.id not in processed_ids]
