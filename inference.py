import os
import json
from typing import List, Optional

from openai import OpenAI

from supplier_env.env import SupplierRiskEnv
from supplier_env.tasks import get_task
from supplier_env.models import Action
from supplier_env.grader import grade_episode

# Required environment variables (Phase 2 rules)
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
API_KEY = os.environ["API_KEY"]

# Environment metadata
BENCHMARK = os.getenv("BENCHMARK", "supplier-risk-openenv")
MAX_STEPS_DEFAULT = 50
SUCCESS_SCORE_THRESHOLD = 0.50


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def smart_policy(obs) -> Action:
    risk_multiplier = 2.5 if obs.week > 10 else 2.0
    target_stock = obs.demand_forecast * risk_multiplier
    gap = max(0, target_stock - obs.inventory)

    order_B = min(int(gap * 0.60), obs.suppliers["B"].max_capacity)
    order_A = min(int(gap * 0.30), obs.suppliers["A"].max_capacity)
    order_C = min(int(gap * 0.10), obs.suppliers["C"].max_capacity)

    return Action(
        order_A=max(0, order_A),
        order_B=max(0, order_B),
        order_C=max(0, order_C),
    )


def get_llm_action(client: OpenAI, obs) -> Action:
    heuristic = smart_policy(obs)

    prompt = f"""
You are a procurement manager AI.

Goal:
- Fulfill demand
- Minimize cost
- Reduce delay and defect risk

Current state:
week={obs.week}
inventory={obs.inventory}
forecast_demand={obs.demand_forecast}
pending_orders={obs.pending_orders}

Recommended safe heuristic:
order_A={heuristic.order_A}
order_B={heuristic.order_B}
order_C={heuristic.order_C}

Return ONLY valid JSON:
{{"order_A": 50, "order_B": 30, "order_C": 40}}
"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert supply chain procurement agent."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=80,
        )

        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)

        return Action(
            order_A=max(0, int(data.get("order_A", heuristic.order_A))),
            order_B=max(0, int(data.get("order_B", heuristic.order_B))),
            order_C=max(0, int(data.get("order_C", heuristic.order_C))),
        )

    except Exception:
        return heuristic


def run_task(task_name: str, client: OpenAI) -> float:
    task = get_task(task_name)
    env = SupplierRiskEnv(task=task, seed=42)

    obs = env.reset()

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        done = False
        max_steps = min(task.max_weeks, MAX_STEPS_DEFAULT)

        for step in range(1, max_steps + 1):
            if done:
                break

            error = None

            try:
                action_obj = get_llm_action(client, obs)
            except Exception as exc:
                error = str(exc)
                action_obj = smart_policy(obs)

            action_str = f'{{"order_A":{action_obj.order_A},"order_B":{action_obj.order_B},"order_C":{action_obj.order_C}}}'

            obs, reward, done, info = env.step(action_obj)

            rewards.append(reward.value)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward.value, done=done, error=error)

        score = grade_episode(env)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    for task_name in ["easy", "medium", "hard"]:
        run_task(task_name, client)


if __name__ == "__main__":
    main()