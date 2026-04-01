"""
Typed models for the Email Triage & Response Environment.

Defines Action, Observation, and State types following the OpenEnv spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class EmailCategory(str, Enum):
    URGENT_BUG = "urgent_bug"
    FEATURE_REQUEST = "feature_request"
    BILLING = "billing"
    GENERAL_INQUIRY = "general_inquiry"
    SPAM = "spam"
    COMPLAINT = "complaint"
    PARTNERSHIP = "partnership"
    INTERNAL = "internal"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResponseTone(str, Enum):
    FORMAL = "formal"
    FRIENDLY = "friendly"
    APOLOGETIC = "apologetic"
    NEUTRAL = "neutral"


# ─── Email Model ──────────────────────────────────────────────────────────────

class Email(BaseModel):
    """A simulated email in the inbox."""
    id: str = Field(..., description="Unique email identifier")
    sender: str = Field(..., description="Sender email address")
    sender_name: str = Field(..., description="Sender display name")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body text")
    timestamp: str = Field(..., description="ISO format timestamp")
    is_reply: bool = Field(default=False, description="Whether this is a reply in a thread")
    thread_id: Optional[str] = Field(default=None, description="Thread ID if part of a conversation")
    # Ground truth (hidden from agent, used by grader)
    _true_category: Optional[EmailCategory] = None
    _true_priority: Optional[Priority] = None
    _expected_tone: Optional[ResponseTone] = None
    _requires_response: bool = True
    _key_points: List[str] = []


# ─── Action (what the agent sends) ───────────────────────────────────────────

class TriageAction(BaseModel):
    """Action taken by the agent on a single email."""
    email_id: str = Field(..., description="ID of the email being acted upon")
    category: str = Field(..., description="Classified category (urgent_bug, feature_request, billing, general_inquiry, spam, complaint, partnership, internal)")
    priority: str = Field(..., description="Assigned priority (critical, high, medium, low)")
    response_draft: str = Field(default="", description="Draft response text (empty if no response needed)")
    response_tone: str = Field(default="neutral", description="Tone of the response (formal, friendly, apologetic, neutral)")
    should_respond: bool = Field(default=True, description="Whether this email requires a response")
    escalate: bool = Field(default=False, description="Whether to escalate to a human supervisor")
    tags: List[str] = Field(default_factory=list, description="Optional tags for organization")


# ─── Observation (what the agent receives) ────────────────────────────────────

class TriageObservation(BaseModel):
    """Observation returned after each step or reset."""
    emails: List[Email] = Field(default_factory=list, description="Current batch of emails to process")
    processed_count: int = Field(default=0, description="Number of emails processed so far")
    total_emails: int = Field(default=0, description="Total emails in this episode")
    current_score: float = Field(default=0.0, description="Running score for this episode")
    feedback: str = Field(default="", description="Feedback on last action taken")
    done: bool = Field(default=False, description="Whether the episode is complete")
    reward: float = Field(default=0.0, description="Reward for the last action")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ─── State (episode metadata) ────────────────────────────────────────────────

class TriageState(BaseModel):
    """Current state of the environment episode."""
    episode_id: str = Field(default="", description="Unique episode identifier")
    task_id: str = Field(default="", description="Current task being executed")
    step_count: int = Field(default=0, description="Number of steps taken")
    max_steps: int = Field(default=0, description="Maximum steps allowed")
    total_reward: float = Field(default=0.0, description="Cumulative reward")
    emails_processed: int = Field(default=0, description="Emails processed so far")
    total_emails: int = Field(default=0, description="Total emails in episode")
    classification_accuracy: float = Field(default=0.0, description="Running classification accuracy")
    response_quality: float = Field(default=0.0, description="Running response quality score")
    done: bool = Field(default=False, description="Whether episode is complete")
