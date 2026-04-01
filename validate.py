#!/usr/bin/env python3
"""
Pre-submission validation script.

Checks all requirements from the hackathon checklist:
1. OpenEnv spec compliance (openenv.yaml, typed models, step/reset/state)
2. Dockerfile exists and has correct structure
3. 3+ tasks with graders producing scores in [0.0, 1.0]
4. Baseline inference reproduces without error
5. Additional endpoints: /baseline, /grader, /tasks
6. Health endpoint returns 200
"""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "✅"
FAIL = "❌"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main():
    print("=" * 64)
    print("  Pre-Submission Validation")
    print("=" * 64)

    # ── 1. Required files ──
    print("\n1. Required Files")
    for f in ["openenv.yaml", "Dockerfile", "README.md", "models.py",
              "inference.py", "requirements.txt", "server/app.py",
              "server/environment.py", "grader.py", "baseline_agent.py"]:
        check(f"File: {f}", os.path.exists(f))

    # ── 2. openenv.yaml structure ──
    print("\n2. OpenEnv YAML Spec")
    import yaml
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    check("Has 'name'", "name" in spec)
    check("Has 'version'", "version" in spec)
    check("Has 'environment'", "environment" in spec)
    check("Has 'environment.tasks'", "tasks" in spec.get("environment", {}))
    tasks = spec.get("environment", {}).get("tasks", [])
    check("Has 3+ tasks", len(tasks) >= 3, f"found {len(tasks)}")
    check("Has 'action_space'", "action_space" in spec.get("environment", {}))
    check("Has 'observation_space'", "observation_space" in spec.get("environment", {}))

    # ── 3. Typed models ──
    print("\n3. Typed Models (Pydantic)")
    from models import TriageAction, TriageObservation, TriageState
    check("TriageAction importable", True)
    check("TriageObservation importable", True)
    check("TriageState importable", True)

    action = TriageAction(email_id="test", category="spam", priority="low")
    check("TriageAction instantiable", action.email_id == "test")

    # ── 4. Environment step/reset/state ──
    print("\n4. Environment API (step/reset/state)")
    from server.environment import EmailTriageEnvironment
    env = EmailTriageEnvironment()

    obs = env.reset(task_id="easy_triage")
    check("reset() returns observation", obs is not None)
    check("reset() has emails", len(obs.emails) > 0)
    check("reset() done=False", not obs.done)

    state = env.state
    check("state has episode_id", bool(state.episode_id))
    check("state step_count=0", state.step_count == 0)

    email = obs.emails[0]
    action = TriageAction(
        email_id=email.id, category="spam", priority="low",
        should_respond=False, response_draft="", response_tone="neutral",
    )
    step_obs = env.step(action)
    check("step() returns observation", step_obs is not None)
    check("step() has reward", isinstance(step_obs.reward, float))
    check("step() reward in [0,1]", 0.0 <= step_obs.reward <= 1.0,
          f"reward={step_obs.reward}")

    # ── 5. Graders ──
    print("\n5. Graders (3 tasks, scores in [0.0, 1.0])")
    from baseline_agent import run_baseline
    env2 = EmailTriageEnvironment()
    baseline_results = run_baseline(env2)

    for task_id in ["easy_triage", "medium_triage", "hard_triage"]:
        score = baseline_results["results"][task_id]["score"]
        check(f"{task_id} score valid", 0.0 <= score <= 1.0,
              f"score={score:.4f}")

    easy = baseline_results["summary"]["easy"]
    hard = baseline_results["summary"]["hard"]
    check("Difficulty gradient (easy > hard)", easy > hard,
          f"easy={easy:.4f}, hard={hard:.4f}")

    # ── 6. Server endpoints ──
    print("\n6. Server Endpoints")
    import uvicorn
    import httpx
    from server.app import app

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=7862, log_level="error")

    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(3)

    client = httpx.Client(base_url="http://localhost:7862", timeout=15)

    r = client.get("/health")
    check("GET /health → 200", r.status_code == 200)

    r = client.post("/reset", json={"task_id": "easy_triage"})
    check("POST /reset → 200", r.status_code == 200)

    obs_data = r.json()
    email_id = obs_data["emails"][0]["id"]
    r = client.post("/step", json={
        "email_id": email_id, "category": "spam", "priority": "low",
        "should_respond": False, "response_draft": "", "response_tone": "neutral",
        "escalate": False, "tags": [],
    })
    check("POST /step → 200", r.status_code == 200)

    r = client.get("/state")
    check("GET /state → 200", r.status_code == 200)

    r = client.get("/tasks")
    check("GET /tasks → 200", r.status_code == 200)
    tasks_data = r.json()
    check("/tasks has 3+ tasks", len(tasks_data["tasks"]) >= 3)
    check("/tasks has action_schema", "action_schema" in tasks_data["tasks"][0])

    r = client.post("/grader")
    check("POST /grader → 200", r.status_code == 200)
    grader_data = r.json()
    check("/grader returns score", "score" in grader_data)

    r = client.post("/baseline")
    check("POST /baseline → 200", r.status_code == 200)
    baseline_data = r.json()
    check("/baseline returns summary", "summary" in baseline_data)

    # ── 7. Dockerfile ──
    print("\n7. Dockerfile")
    with open("Dockerfile") as f:
        dockerfile = f.read()
    check("Has FROM", "FROM" in dockerfile)
    check("Has EXPOSE", "EXPOSE" in dockerfile)
    check("Has CMD", "CMD" in dockerfile)
    check("Has HEALTHCHECK", "HEALTHCHECK" in dockerfile)

    # ── Summary ──
    print("\n" + "=" * 64)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    all_passed = passed == total
    status = PASS if all_passed else FAIL
    print(f"  {status} {passed}/{total} checks passed")
    if not all_passed:
        print("\n  Failed checks:")
        for name, ok in results:
            if not ok:
                print(f"    {FAIL} {name}")
    print("=" * 64)

    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "pyyaml", "--break-system-packages", "-q"])
        import yaml

    sys.exit(main())
