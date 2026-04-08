import asyncio
import os
import textwrap
import json
import inspect
from typing import Any, List, Optional
from openai import OpenAI

# 1. IMPORT YOUR ACTUAL ENVIRONMENT HERE
from env import ResilientOpsAction, ResilientOpsEnv

IMAGE_NAME = os.getenv("IMAGE_NAME") 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini") # Use a fast/cheap model for baseline

KNOWN_TASKS = ("minor-delay", "port-strike", "network-collapse")
TASK_NAME = os.getenv("MY_ENV_V4_TASK")
TASKS_CSV = os.getenv("MY_ENV_V4_TASKS", ",".join(KNOWN_TASKS))
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "resilientops")

MAX_STEPS = 14
TEMPERATURE = 0.2
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5 

MAX_TOTAL_REWARD = 14.0 

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Supply Chain Manager. Your goal is to maximize profit over a 14-day crisis.
    You will receive JSON observations of inventory, transit shipments, and active crises.
    You must output a raw JSON action with exactly these keys:
    {"supplier_id": "str", "product_id": "str", "quantity": int, "transport_mode": "STANDARD" or "AIR"}
    Reply with ONLY valid JSON. No markdown, no quotes, no explanations.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def resolve_tasks() -> List[str]:
    """Resolve evaluation task set, defaulting to all known tasks."""

    if TASK_NAME:
        selected = [TASK_NAME]
    else:
        selected = [t.strip() for t in TASKS_CSV.split(",") if t.strip()]

    deduped: List[str] = []
    for task in selected:
        if task not in KNOWN_TASKS:
            raise ValueError(
                f"Unknown task '{task}'. Supported tasks: {', '.join(KNOWN_TASKS)}"
            )
        if task not in deduped:
            deduped.append(task)

    if not deduped:
        raise ValueError("No tasks selected for inference run.")

    return deduped

def build_user_prompt(step: int, obs: dict, last_reward: float) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}
        Current State: {json.dumps(obs)}
        Last reward: {last_reward:.2f}
        Provide your next action as JSON.
        """
    ).strip()

def build_fallback_action(step: int, obs: dict) -> str:
    """Deterministic policy used when model calls are unavailable."""

    demand = obs.get("daily_demand", {})
    inventory = obs.get("warehouse_inventory", {})
    alerts = " ".join(obs.get("active_crisis_alerts", []))

    microchip_demand = 0
    battery_demand = 0
    microchip_inventory = 0
    battery_inventory = 0

    for warehouse in demand.values():
        if isinstance(warehouse, dict):
            microchip_demand += int(warehouse.get("microchip", 0) or 0)
            battery_demand += int(warehouse.get("battery", 0) or 0)

    for warehouse in inventory.values():
        if isinstance(warehouse, dict):
            microchip_inventory += int(warehouse.get("microchip", 0) or 0)
            battery_inventory += int(warehouse.get("battery", 0) or 0)

    product_id = "microchip" if microchip_demand >= battery_demand else "battery"

    if product_id == "microchip":
        supplier_id = "sup_beta" if step % 3 == 0 else "sup_alpha"
        product_demand = microchip_demand
        product_inventory = microchip_inventory
    else:
        supplier_id = "sup_gamma" if step % 2 == 0 else "sup_alpha"
        product_demand = battery_demand
        product_inventory = battery_inventory

    quantity = int(max(0, min(140, (product_demand * 2) - int(product_inventory * 0.30))))

    # Keep ordering aggressive when network collapse starts.
    if "network-collapse" in alerts and step >= 5:
        quantity = max(quantity, 100)

    transport_mode = "STANDARD"
    if "port-strike" in alerts and 4 <= step <= 9:
        transport_mode = "AIR"

    return json.dumps(
        {
            "supplier_id": supplier_id,
            "product_id": product_id,
            "quantity": quantity,
            "transport_mode": transport_mode,
        }
    )


def get_model_action(client: Optional[OpenAI], step: int, obs: dict, last_reward: float) -> str:
    if client is None:
        return build_fallback_action(step, obs)

    user_prompt = build_user_prompt(step, obs, last_reward)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            response_format={ "type": "json_object" } # Forces JSON output
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return build_fallback_action(step, obs)


async def maybe_await(value):
    """Handle both sync and async environment methods."""

    if inspect.isawaitable(value):
        return await value
    return value

async def run_episode(task_name: str, client: Optional[OpenAI]) -> None:
    """Run one task episode and emit required structured logs."""

    env = ResilientOpsEnv()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await maybe_await(env.reset(task_id=task_name))
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs_dict = result.observation.model_dump()
            action_json_str = get_model_action(client, step, obs_dict, last_reward)

            try:
                action_dict = json.loads(action_json_str)
                action_obj = ResilientOpsAction(**action_dict)
                result = await maybe_await(env.step(action_obj))
                reward = result.reward.value or 0.0
                error = None
            except Exception as e:
                reward = -1.0
                error = str(e)
                result = await maybe_await(
                    env.step(
                        ResilientOpsAction(
                            supplier_id="sup_alpha",
                            product_id="microchip",
                            quantity=0,
                            transport_mode="STANDARD",
                        )
                    )
                )

            done = result.done
            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            action_log = action_json_str.replace('\n', '').replace('\r', '')
            log_step(step=step, action=action_log, reward=reward, done=done, error=error)

            if done:
                score = float(result.info.get("final_score", 0.0))
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await maybe_await(env.close())
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client: Optional[OpenAI] = None
    if API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception as exc:
            print(f"[DEBUG] OpenAI client init failed: {exc}", flush=True)
            print("[DEBUG] Falling back to deterministic policy", flush=True)
    else:
        print("[DEBUG] Missing API key; falling back to deterministic policy", flush=True)

    tasks = resolve_tasks()
    for task_name in tasks:
        await run_episode(task_name=task_name, client=client)

if __name__ == "__main__":
    asyncio.run(main())