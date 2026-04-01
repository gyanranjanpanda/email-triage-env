"""
Email dataset generator for the Email Triage Environment.

Generates realistic email scenarios with ground-truth labels for
classification, priority, response requirements, and expected tone.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from models import Email, EmailCategory, Priority, ResponseTone


def _ts(offset_hours: int = 0) -> str:
    """Generate an ISO timestamp offset from now."""
    return (datetime.utcnow() - timedelta(hours=offset_hours)).isoformat() + "Z"


# ─── Email templates ─────────────────────────────────────────────────────────
# Each tuple: (sender, sender_name, subject, body, category, priority, tone, requires_response, key_points)

EASY_EMAILS: List[Dict] = [
    {
        "sender": "mike.chen@bigcorp.com",
        "sender_name": "Mike Chen",
        "subject": "URGENT: Production database is down!",
        "body": "Hi team,\n\nOur production database cluster went offline at 3:42 AM EST. All customer-facing APIs are returning 500 errors. We need immediate attention. Over 10,000 users are affected.\n\nError log attached. Please escalate ASAP.\n\nMike Chen\nSr. DevOps Engineer",
        "category": EmailCategory.URGENT_BUG,
        "priority": Priority.CRITICAL,
        "tone": ResponseTone.FORMAL,
        "requires_response": True,
        "key_points": ["acknowledge urgency", "confirm escalation", "provide timeline"],
    },
    {
        "sender": "spam@win-prizes-now.xyz",
        "sender_name": "Prize Committee",
        "subject": "🎉 You've WON $1,000,000! Claim NOW!!!",
        "body": "CONGRATULATIONS!!!\n\nYou have been selected as our GRAND PRIZE WINNER! Click here to claim your $1,000,000 prize. Act now before it expires! Send us your bank details immediately.\n\nThis is NOT a scam. Limited time offer!!!",
        "category": EmailCategory.SPAM,
        "priority": Priority.LOW,
        "tone": ResponseTone.NEUTRAL,
        "requires_response": False,
        "key_points": [],
    },
    {
        "sender": "jane.smith@example.com",
        "sender_name": "Jane Smith",
        "subject": "Question about pricing plans",
        "body": "Hello,\n\nI'm interested in upgrading from the free tier to a paid plan. Could you explain the difference between the Pro and Enterprise tiers? Specifically, I'd like to know about API rate limits and support SLAs.\n\nThanks,\nJane",
        "category": EmailCategory.BILLING,
        "priority": Priority.MEDIUM,
        "tone": ResponseTone.FRIENDLY,
        "requires_response": True,
        "key_points": ["explain plan differences", "mention API limits", "offer to help"],
    },
    {
        "sender": "alex.dev@startup.io",
        "sender_name": "Alex Developer",
        "subject": "Feature request: Dark mode support",
        "body": "Hey there!\n\nLove your product. One thing that would make my workflow so much better is dark mode. I work late nights and the bright UI is tough on my eyes. Would be great if you could add this to the roadmap.\n\nCheers,\nAlex",
        "category": EmailCategory.FEATURE_REQUEST,
        "priority": Priority.LOW,
        "tone": ResponseTone.FRIENDLY,
        "requires_response": True,
        "key_points": ["thank for feedback", "acknowledge request", "mention roadmap"],
    },
    {
        "sender": "sarah.vp@megacorp.com",
        "sender_name": "Sarah Johnson",
        "subject": "Partnership opportunity - MegaCorp x YourCompany",
        "body": "Dear Team,\n\nI'm the VP of Strategic Partnerships at MegaCorp. We've been impressed with your product and would like to explore a potential integration partnership. Our platform serves 50M+ users and we believe there's strong synergy.\n\nWould love to set up a call next week to discuss.\n\nBest regards,\nSarah Johnson\nVP Strategic Partnerships, MegaCorp",
        "category": EmailCategory.PARTNERSHIP,
        "priority": Priority.HIGH,
        "tone": ResponseTone.FORMAL,
        "requires_response": True,
        "key_points": ["express interest", "suggest meeting", "professional tone"],
    },
]

MEDIUM_EMAILS: List[Dict] = [
    {
        "sender": "frustrated.user@gmail.com",
        "sender_name": "Tom Wilson",
        "subject": "This is the third time I've reported this issue",
        "body": "I'm extremely frustrated. For the third time this month, my data exports are failing silently. No error messages, no notifications — just missing data. I've already contacted support twice and was told it would be 'fixed soon.' I'm a paying customer on your Enterprise plan and this level of service is unacceptable.\n\nIf this isn't resolved within 24 hours, I'll be forced to evaluate competitors.\n\nTom Wilson\nCTO, Wilson Analytics",
        "category": EmailCategory.COMPLAINT,
        "priority": Priority.HIGH,
        "tone": ResponseTone.APOLOGETIC,
        "requires_response": True,
        "key_points": ["apologize sincerely", "acknowledge repeated issue", "provide concrete timeline", "escalate internally"],
    },
    {
        "sender": "newsletter@techdigest.com",
        "sender_name": "Tech Digest Weekly",
        "subject": "This week in AI: New breakthroughs and trends",
        "body": "Tech Digest Weekly Newsletter\n\n1. GPT-5 rumors intensify\n2. New EU AI regulations\n3. Open source model benchmarks\n\nClick to read more. Unsubscribe at any time.",
        "category": EmailCategory.SPAM,
        "priority": Priority.LOW,
        "tone": ResponseTone.NEUTRAL,
        "requires_response": False,
        "key_points": [],
    },
    {
        "sender": "recruiter@talent.com",
        "sender_name": "Lisa Park",
        "subject": "Exciting opportunity at a Series B startup",
        "body": "Hi there,\n\nI came across your profile and thought you'd be a great fit for a Senior Engineer role at one of my client companies. They've just raised $50M Series B and are scaling their team.\n\nWould you be open to a 15-min chat?\n\nBest,\nLisa Park\nTalent Acquisition",
        "category": EmailCategory.GENERAL_INQUIRY,
        "priority": Priority.LOW,
        "tone": ResponseTone.NEUTRAL,
        "requires_response": True,
        "key_points": ["polite decline or acknowledgment"],
    },
    {
        "sender": "cto@ourcompany.com",
        "sender_name": "David Kim",
        "subject": "Re: Q2 roadmap priorities",
        "body": "Team,\n\nAfter reviewing the Q2 proposals, I'd like us to prioritize:\n1. Performance optimization (P0)\n2. Mobile app v2 (P1)\n3. Analytics dashboard revamp (P2)\n\nPlease update your sprint plans accordingly by Friday. Let me know if there are any blockers.\n\n—David",
        "category": EmailCategory.INTERNAL,
        "priority": Priority.HIGH,
        "tone": ResponseTone.FORMAL,
        "requires_response": True,
        "key_points": ["acknowledge priorities", "confirm timeline", "flag any blockers"],
        "is_reply": True,
        "thread_id": "thread-q2-roadmap",
    },
    {
        "sender": "security@bank-alert.com",
        "sender_name": "Bank Security",
        "subject": "Suspicious activity detected on your account",
        "body": "Dear Valued Customer,\n\nWe detected suspicious activity on your account. Please verify your identity by clicking the link below and entering your credentials:\n\nhttp://bank-security-verify.suspicious-domain.com/login\n\nFailure to verify within 24 hours will result in account suspension.\n\nBank Security Team",
        "category": EmailCategory.SPAM,
        "priority": Priority.LOW,
        "tone": ResponseTone.NEUTRAL,
        "requires_response": False,
        "key_points": [],
    },
    {
        "sender": "beta.tester@gmail.com",
        "sender_name": "Raj Patel",
        "subject": "Bug: Intermittent 403 errors on API v2 endpoint",
        "body": "Hi Support,\n\nI'm experiencing intermittent 403 Forbidden errors when hitting the /api/v2/analytics endpoint. It works fine ~80% of the time but randomly fails. I've confirmed my API key is valid and my rate limits aren't exceeded.\n\nHere are the request headers and response details:\n- Endpoint: GET /api/v2/analytics?range=7d\n- Auth: Bearer token (valid, checked)\n- Response: 403 ~20% of requests\n- Started: 2 days ago\n\nHappy to provide more logs if needed.\n\nRaj",
        "category": EmailCategory.URGENT_BUG,
        "priority": Priority.HIGH,
        "tone": ResponseTone.FRIENDLY,
        "requires_response": True,
        "key_points": ["thank for detailed report", "acknowledge issue", "request additional logs", "provide ticket number"],
    },
    {
        "sender": "accounting@ourcompany.com",
        "sender_name": "Finance Team",
        "subject": "Reminder: Submit expense reports by Friday",
        "body": "Hi all,\n\nFriendly reminder that all March expense reports are due by this Friday at 5 PM. Please submit through the finance portal.\n\nQuestions? Reply to this email.\n\nFinance Team",
        "category": EmailCategory.INTERNAL,
        "priority": Priority.MEDIUM,
        "tone": ResponseTone.NEUTRAL,
        "requires_response": False,
        "key_points": [],
    },
]

HARD_EMAILS: List[Dict] = [
    {
        "sender": "legal@competitor.com",
        "sender_name": "Legal Department",
        "subject": "Notice of potential IP infringement",
        "body": "Dear Sir/Madam,\n\nIt has come to our attention that certain features in your product may infringe upon patents held by our company (US Patent No. 11,234,567 and US Patent No. 11,345,678). We request that you review these patents and provide a response within 30 days.\n\nThis letter is not intended as a formal cease and desist but rather as a good-faith attempt to resolve this matter amicably.\n\nSincerely,\nLegal Department\nCompetitor Inc.",
        "category": EmailCategory.GENERAL_INQUIRY,
        "priority": Priority.CRITICAL,
        "tone": ResponseTone.FORMAL,
        "requires_response": True,
        "key_points": ["acknowledge receipt", "do NOT admit liability", "state will review with legal team", "professional tone", "escalate to legal"],
    },
    {
        "sender": "longtime.customer@enterprise.com",
        "sender_name": "Maria Garcia",
        "subject": "Considering downgrade - disappointed with recent changes",
        "body": "Hi,\n\nI've been a loyal Enterprise customer for 4 years now, paying $2,400/month. The recent UI redesign has significantly impacted our team's productivity. Several features we relied on daily were removed or moved behind additional clicks.\n\nAdditionally, the last two 'improvements' to the API actually broke our integration twice. Our engineers spent 40+ hours fixing things each time.\n\nI'd like to discuss either:\n1. A rollback option for our account\n2. A significant discount to compensate for the disruption\n3. Or I'll need to begin our migration to [Competitor]\n\nI'd prefer to stay, but I need to see that our concerns are heard.\n\nMaria Garcia\nHead of Engineering, Enterprise Solutions Inc.",
        "category": EmailCategory.COMPLAINT,
        "priority": Priority.CRITICAL,
        "tone": ResponseTone.APOLOGETIC,
        "requires_response": True,
        "key_points": ["empathize sincerely", "acknowledge specific issues", "do NOT promise rollback without checking", "offer to connect with account manager", "retention focus", "escalate"],
    },
    {
        "sender": "press@techmedia.com",
        "sender_name": "Jordan Lee",
        "subject": "Media inquiry: Data breach rumors",
        "body": "Hi,\n\nI'm a reporter with TechMedia. We've received an anonymous tip suggesting your company experienced a data breach affecting user payment information. We plan to publish a story tomorrow morning.\n\nCould you provide a comment or statement? Any response received by 6 PM today will be included in our coverage.\n\nThank you,\nJordan Lee\nSenior Tech Reporter, TechMedia",
        "category": EmailCategory.GENERAL_INQUIRY,
        "priority": Priority.CRITICAL,
        "tone": ResponseTone.FORMAL,
        "requires_response": True,
        "key_points": ["do NOT confirm or deny breach", "state reviewing with security team", "escalate to PR/legal immediately", "professional measured tone"],
    },
    {
        "sender": "ciso@ourcompany.com",
        "sender_name": "Robert Chen",
        "subject": "CONFIDENTIAL: Security incident response",
        "body": "CONFIDENTIAL - DO NOT FORWARD\n\nTeam,\n\nWe've detected unauthorized access to our staging environment. Initial analysis suggests it was accessed via a compromised service account. No evidence of production data exposure yet, but investigation is ongoing.\n\nImmediate actions needed:\n1. Rotate all staging credentials\n2. Review access logs for the past 72 hours\n3. Do NOT discuss this outside the incident response team\n\nWar room at 10 AM. Calendar invite incoming.\n\n—Robert Chen\nCISO",
        "category": EmailCategory.INTERNAL,
        "priority": Priority.CRITICAL,
        "tone": ResponseTone.FORMAL,
        "requires_response": True,
        "key_points": ["acknowledge confidentiality", "confirm attendance", "brief response only"],
        "is_reply": False,
    },
    {
        "sender": "investor@vcfund.com",
        "sender_name": "Patricia Wong",
        "subject": "Follow-up: Series C discussion",
        "body": "Hi Team,\n\nGreat meeting last week. We've discussed internally and our investment committee is interested in leading your Series C at a $500M pre-money valuation.\n\nA few items we'd like to discuss:\n1. Updated financial projections for FY2026\n2. Customer retention metrics (especially enterprise segment)\n3. Competitive moat analysis\n\nCould we schedule a follow-up for next Tuesday or Wednesday?\n\nBest,\nPatricia Wong\nGeneral Partner, VC Fund Capital",
        "category": EmailCategory.PARTNERSHIP,
        "priority": Priority.HIGH,
        "tone": ResponseTone.FORMAL,
        "requires_response": True,
        "key_points": ["express enthusiasm", "confirm availability", "note preparation items", "professional and polished"],
    },
    {
        "sender": "angry.user@hotmail.com",
        "sender_name": "Dave Thompson",
        "subject": "Your product deleted all my data!!!",
        "body": "I AM FURIOUS. I logged in this morning and ALL of my projects are GONE. 3 years of work just VANISHED. I've been paying $49/month for your 'reliable' service and this is what I get??\n\nI want:\n1. My data restored IMMEDIATELY\n2. A full refund for the last 12 months\n3. An explanation of how this happened\n\nIf I don't hear back within 2 hours I'm posting this on Twitter and contacting my lawyer.\n\nDave Thompson",
        "category": EmailCategory.URGENT_BUG,
        "priority": Priority.CRITICAL,
        "tone": ResponseTone.APOLOGETIC,
        "requires_response": True,
        "key_points": ["calm and empathetic", "take data loss seriously", "do NOT promise refund without authorization", "escalate to engineering", "provide interim update timeline"],
    },
    {
        "sender": "hr@ourcompany.com",
        "sender_name": "HR Department",
        "subject": "Mandatory: Updated code of conduct - please acknowledge",
        "body": "Dear Team Member,\n\nAs part of our annual compliance update, please review and acknowledge the updated Code of Conduct document. This includes new sections on:\n- AI usage guidelines\n- Remote work policies\n- Data handling procedures\n\nPlease complete by end of month. Link: [internal portal]\n\nHR Department",
        "category": EmailCategory.INTERNAL,
        "priority": Priority.MEDIUM,
        "tone": ResponseTone.NEUTRAL,
        "requires_response": False,
        "key_points": [],
    },
    {
        "sender": "confused.user@gmail.com",
        "sender_name": "Alex Nguyen",
        "subject": "Can't figure out the API authentication",
        "body": "Hi,\n\nI've been trying to integrate your API for 2 days now but keep getting authentication errors. I've tried:\n- Using the API key directly in the header\n- Using Bearer token format\n- Regenerating the key\n\nYour docs say to use 'X-API-Key' header but that returns 401. The example in the getting started guide seems outdated.\n\nI'm building a school project and the deadline is tomorrow. Any help would be amazing.\n\nThanks,\nAlex",
        "category": EmailCategory.GENERAL_INQUIRY,
        "priority": Priority.MEDIUM,
        "tone": ResponseTone.FRIENDLY,
        "requires_response": True,
        "key_points": ["helpful and patient", "provide correct auth method", "acknowledge docs may be outdated", "wish luck on project"],
    },
]


def generate_email_batch(
    difficulty: str,
    batch_size: int = 5,
    seed: int | None = None,
) -> Tuple[List[Email], List[Dict]]:
    """
    Generate a batch of emails with ground truth labels.

    Returns:
        (emails, ground_truths) where ground_truths is a list of dicts
        with the true labels for grading.
    """
    if seed is not None:
        random.seed(seed)

    if difficulty == "easy":
        pool = EASY_EMAILS
    elif difficulty == "medium":
        pool = EASY_EMAILS + MEDIUM_EMAILS
    else:  # hard
        pool = EASY_EMAILS + MEDIUM_EMAILS + HARD_EMAILS

    # Select emails, allowing repeats if batch_size > pool
    selected = random.sample(pool, min(batch_size, len(pool)))
    if batch_size > len(pool):
        selected += random.choices(pool, k=batch_size - len(pool))

    random.shuffle(selected)

    emails = []
    ground_truths = []

    for i, template in enumerate(selected):
        email_id = f"email-{i+1:03d}"
        email = Email(
            id=email_id,
            sender=template["sender"],
            sender_name=template["sender_name"],
            subject=template["subject"],
            body=template["body"],
            timestamp=_ts(offset_hours=random.randint(0, 48)),
            is_reply=template.get("is_reply", False),
            thread_id=template.get("thread_id"),
        )
        emails.append(email)

        ground_truths.append({
            "email_id": email_id,
            "category": template["category"].value,
            "priority": template["priority"].value,
            "tone": template["tone"].value,
            "requires_response": template["requires_response"],
            "key_points": template.get("key_points", []),
        })

    return emails, ground_truths
