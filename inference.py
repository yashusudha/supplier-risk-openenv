import os
import json

from openai import OpenAI

from supplier_env.env import SupplierRiskEnv
from supplier_env.tasks import get_task
from supplier_env.models import Action
from supplier_env.grader import grade_episode


def log_start(task_name: str):
    print(f"[START] task={task_name}")


def log_step(t: int, obs, action: Action, reward_value: float, done: bool):
    print(
        f"[STEP] t={t} inventory={obs.inventory} forecast={obs.demand_forecast} "
        f"action=({action.order_A},{action.order_B},{action.order_C}) "
        f"reward={reward_value:.2f} done={done}"
    )


def log_end(task_name: str, score: float):
    print(f"[END] task={task_name} score={score:.4f}")


def get_llm_action(client: OpenAI, model: str, obs) -> Action:
    """
    Uses OpenAI client to generate action.
    Output must be JSON strictly.
    """

    prompt = f"""
You are a procurement manager AI.
Decide order quantities from 3 suppliers A, B, C.

Goal:
- Avoid stockouts
- Avoid excess inventory
- Reduce ordering cost
- Reduce defect risk and delay risk

Current State:
week={obs.week}
inventory={obs.inventory}
forecast_demand={obs.demand_forecast}
pending_orders={obs.pending_orders}

Supplier Info:
A: cost={obs.suppliers['A'].cost_per_unit}, delay_prob={obs.suppliers['A'].delay_probability}, defect_prob={obs.suppliers['A'].defect_probability}, max_capacity={obs.suppliers['A'].max_capacity}
B: cost={obs.suppliers['B'].cost_per_unit}, delay_prob={obs.suppliers['B'].delay_probability}, defect_prob={obs.suppliers['B'].defect_probability}, max_capacity={obs.suppliers['B'].max_capacity}
C: cost={obs.suppliers['C'].cost_per_unit}, delay_prob={obs.suppliers['C'].delay_probability}, defect_prob={obs.suppliers['C'].defect_probability}, max_capacity={obs.suppliers['C'].max_capacity}

Return ONLY valid JSON like:
{{"order_A": 50, "order_B": 20, "order_C": 40}}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert supply chain procurement agent."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    text = resp.choices[0].message.content.strip()

    try:
        data = json.loads(text)
        return Action(
            order_A=int(data.get("order_A", 0)),
            order_B=int(data.get("order_B", 0)),
            order_C=int(data.get("order_C", 0)),
        )
    except Exception:
        # fallback safe action
        return Action(order_A=50, order_B=30, order_C=50)


def run_task(task_name: str, client: OpenAI, model: str) -> float:
    task = get_task(task_name)
    env = SupplierRiskEnv(task=task, seed=42)

    obs = env.reset()
    log_start(task_name)

    t = 0
    done = False
    while not done:
        t += 1
        action = get_llm_action(client, model, obs)
        obs, reward, done, info = env.step(action)
        log_step(t, obs, action, reward.value, done)

    score = grade_episode(env)
    log_end(task_name, score)

    return score


def main():
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    if not api_base_url or not model_name or not hf_token:
        raise RuntimeError(
            "Missing required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN"
        )

    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token,
    )

    scores = []
    for task_name in ["easy", "medium", "hard"]:
        score = run_task(task_name, client, model_name)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    print(f"[END] task=all average_score={avg_score:.4f}")


if __name__ == "__main__":
    main()