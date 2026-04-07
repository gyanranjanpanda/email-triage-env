---
title: Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Email Triage & Response Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement learning environment that simulates real-world email triage and response tasks. AI agents learn to classify, prioritize, and respond to emails across increasing difficulty levels.

## Why Email Triage?

Email triage is a genuine, high-impact business task performed millions of times daily by customer support teams, executive assistants, and operations staff. Unlike toy environments, this task requires:

- **Multi-dimensional classification** — category, priority, and urgency assessment
- **Nuanced decision-making** — should the agent respond, escalate, or ignore?
- **Language generation** — drafting appropriate responses with correct tone
- **Contextual reasoning** — legal notices need different handling than feature requests
- **Partial credit** — real-world performance isn't binary; a nearly-correct classification still has value

## Environment Overview

The agent receives a batch of simulated emails and must process each one by:

1. **Classifying** the email into one of 8 categories
2. **Assigning priority** (critical / high / medium / low)
3. **Deciding** whether to respond, ignore, or escalate
4. **Drafting** an appropriate response with the right tone
5. **Tagging** for organization

### Action Space

| Field            | Type     | Required | Description                                |
|------------------|----------|----------|--------------------------------------------|
| `email_id`       | string   | ✅       | ID of the email being processed            |
| `category`       | string   | ✅       | One of: `urgent_bug`, `feature_request`, `billing`, `general_inquiry`, `spam`, `complaint`, `partnership`, `internal` |
| `priority`       | string   | ✅       | One of: `critical`, `high`, `medium`, `low`|
| `response_draft` | string   | ❌       | Draft response text                        |
| `response_tone`  | string   | ❌       | One of: `formal`, `friendly`, `apologetic`, `neutral` |
| `should_respond` | boolean  | ❌       | Whether this email requires a response     |
| `escalate`       | boolean  | ❌       | Whether to escalate to a supervisor        |
| `tags`           | string[] | ❌       | Optional organizational tags               |

### Observation Space

| Field             | Type    | Description                              |
|-------------------|---------|------------------------------------------|
| `emails`          | Email[] | List of unprocessed emails               |
| `processed_count` | int     | Number of emails processed so far        |
| `total_emails`    | int     | Total emails in this episode             |
| `current_score`   | float   | Running average score                    |
| `feedback`        | string  | Feedback on the last action              |
| `done`            | bool    | Whether the episode is complete          |
| `reward`          | float   | Reward for the last action (0.0 – 1.0)  |
| `metadata`        | object  | Additional episode metadata              |

## Tasks (Easy → Medium → Hard)

### Task 1: `easy_triage` (Easy)
- **5 emails** — obvious spam, clear bug reports, straightforward inquiries
- Grading weighted toward classification accuracy (50%)
- Baseline score: **0.9636**

### Task 2: `medium_triage` (Medium)
- **7 emails** — phishing mixed with legitimate mail, angry complaints, ambiguous internal memos
- Grading shifts toward response quality and tone (35%)
- Baseline score: **0.9234**

### Task 3: `hard_triage` (Hard)
- **10 emails** — legal threats, press inquiries about data breaches, security incidents, investor communications, high-value customer retention
- Grading heavily weights content quality and escalation decisions (45%)
- Baseline score: **0.7189**

## Reward Function

Each action is graded across 5 dimensions with difficulty-dependent weights:

| Component           | Easy | Medium | Hard | Description |
|----------------------|------|--------|------|-------------|
| Classification       | 50%  | 30%    | 20%  | Category + priority accuracy |
| Response decision    | 25%  | 20%    | 15%  | Respond vs. ignore correctness |
| Tone                 | 10%  | 15%    | 20%  | Response tone appropriateness |
| Content quality      | 10%  | 20%    | 25%  | Key points coverage in response |
| Escalation           |  5%  | 15%    | 20%  | Correct escalation decisions |

Partial credit is awarded for close-but-not-perfect answers (e.g., classifying a complaint as an urgent bug still earns partial credit).

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the server

```bash
python server/app.py
# Server runs at http://localhost:7860
```

### 3. Interact with the API

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_triage"}'

# Process an email
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "email_id": "email-001",
    "category": "spam",
    "priority": "low",
    "should_respond": false,
    "response_draft": "",
    "response_tone": "neutral",
    "escalate": false
  }'

# Check state
curl http://localhost:7860/state

# Get grader score
curl -X POST http://localhost:7860/grader

# Run full baseline
curl -X POST http://localhost:7860/baseline
```

### 4. Run the baseline inference script

```bash
# Direct mode (no server needed)
python inference.py

# Against a running server
python inference.py --server http://localhost:7860
```

## API Endpoints

| Endpoint    | Method | Description                                          |
|-------------|--------|------------------------------------------------------|
| `/health`   | GET    | Health check — returns 200 + status                  |
| `/reset`    | POST   | Start new episode with `{task_id, seed}`             |
| `/step`     | POST   | Execute triage action, returns observation + reward   |
| `/state`    | GET    | Current episode state and metadata                   |
| `/tasks`    | GET    | List tasks with action schema                        |
| `/grader`   | POST   | Grade completed episode, returns score 0.0–1.0       |
| `/baseline` | POST   | Run baseline agent on all tasks, returns all scores   |

## Docker Deployment

```bash
# Build
docker build -t email-triage-env .

# Run
docker run -p 7860:7860 email-triage-env

# Test
curl http://localhost:7860/health
```

## Deploy to Hugging Face Spaces

```bash
# Using OpenEnv CLI
openenv push --repo-id your-username/email-triage-env

# Or manually push to an HF Space with Docker SDK
```

## Project Structure

```
email_triage_env/
├── __init__.py
├── models.py              # Typed Action, Observation, State models
├── email_data.py           # Email dataset generator with ground truth
├── grader.py               # Multi-dimensional grading logic
├── baseline_agent.py       # Heuristic keyword-matching agent
├── inference.py            # Standalone baseline inference script
├── openenv.yaml            # OpenEnv manifest
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container image definition
├── README.md               # This file
└── server/
    ├── __init__.py
    ├── app.py              # FastAPI server with all endpoints
    └── environment.py      # Core environment logic (step/reset/state)
```

## Baseline Agent

The included baseline uses simple keyword matching:
- Scans email text for category-specific keywords (e.g., "production down" → urgent_bug)
- Applies priority based on severity signals
- Uses template responses per category
- Escalates all critical-priority items

This achieves reasonable scores on easy tasks but struggles with nuanced hard scenarios (legal notices, press inquiries), leaving significant room for improvement by learning agents.

## Reproducibility

All tasks use fixed random seeds, ensuring identical email batches across runs:
- `easy_triage`: seed=42
- `medium_triage`: seed=123
- `hard_triage`: seed=456

Running `python inference.py` will always produce:
```
easy   : 0.9636
medium : 0.9234
hard   : 0.7189
```

## License

BSD-3-Clause
