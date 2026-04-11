"""
Baseline inference script for ResilientOps v2.

Uses OpenAI-compatible API to run an LLM agent against all 3 tasks.
Falls back to a deterministic heuristic policy when the API is unavailable.

Required environment variables:
  API_BASE_URL  - LLM API endpoint
  MODEL_NAME    - model identifier
  HF_TOKEN      - API key (or OPENAI_API_KEY)
"""

import asyncio
import inspect
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

from env import COMPONENTS, WAREHOUSES, ResilientOpsAction, ResilientOpsEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

KNOWN_TASKS = ("minor-delay", "port-strike", "cascading-failures")
TASK_NAME = os.getenv("MY_ENV_V2_TASK") or os.getenv("MY_ENV_V4_TASK")
TASKS_CSV = os.getenv("MY_ENV_V2_TASKS") or os.getenv("MY_ENV_V4_TASKS", ",".join(KNOWN_TASKS))
BENCHMARK = os.getenv("MY_ENV_V2_BENCHMARK") or os.getenv("MY_ENV_V4_BENCHMARK", "resilientops-v2")

MAX_STEPS = 14
TEMPERATURE = 0.2
MAX_TOKENS = 500
SUCCESS_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rstr}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Supply Chain Operations Manager for NovaTech Electronics.

    You manage 3 warehouses (na_hub, eu_hub, apac_hub) and 3 suppliers:
    - sup_alpha: sells microchip + battery, 2-day standard lead, capacity 180/day
    - sup_beta: sells microchip ONLY, 3-day standard lead, capacity 120/day
    - sup_gamma: sells battery ONLY, 3-day standard lead, capacity 120/day

    PRODUCTS & BILL OF MATERIALS:
    - microchip: component, sells for $30/unit
    - battery: component, sells for $25/unit
    - device: assembled from 1 microchip + 1 battery, sells for $85/unit (assembly cost $5)

    Assembling devices is more profitable than selling raw components.

    CRITICAL: Read inbox messages carefully.
    You must output a single JSON object with these exact fields:
    {
      "supplier_id": "sup_alpha|sup_beta|sup_gamma|empty string",
      "product_id": "microchip|battery|empty string",
      "quantity": 0,
      "transport_mode": "STANDARD|AIR",
      "transfer_product_id": "microchip|battery|empty string",
      "transfer_quantity": 0,
      "transfer_from": "warehouse_id|empty string",
      "transfer_to": "warehouse_id|empty string",
      "assemble_warehouse": "warehouse_id|empty string",
      "assemble_quantity": 0,
      "priority": "enterprise|retail|balanced"
    }

    Leave optional fields as empty strings or 0 if unused.
    Reply with only valid JSON.
"""
).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_tasks() -> List[str]:
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


def build_user_prompt(step: int, obs: Dict[str, Any], last_reward: float) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}
        Observation JSON: {json.dumps(obs)}
        Last reward: {last_reward:.2f}
        Return the next action JSON.
        """
    ).strip()


def _sum_component_totals(table: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    totals = {p: 0 for p in COMPONENTS}
    for wh in WAREHOUSES:
        wh_map = table.get(wh, {})
        for p in COMPONENTS:
            totals[p] += int(wh_map.get(p, 0) or 0)
    return totals


def _default_action() -> Dict[str, Any]:
    return {
        "supplier_id": "",
        "product_id": "",
        "quantity": 0,
        "transport_mode": "STANDARD",
        "transfer_product_id": "",
        "transfer_quantity": 0,
        "transfer_from": "",
        "transfer_to": "",
        "assemble_warehouse": "",
        "assemble_quantity": 0,
        "priority": "balanced",
    }


# ---------------------------------------------------------------------------
# Fallback deterministic policy
# ---------------------------------------------------------------------------

def fallback_action(task: str, step: int, obs: Dict[str, Any]) -> str:
    """Heuristic policy that uses order, transfer, and assembly levers."""

    inventory = obs.get("warehouse_inventory", {})
    demand = obs.get("daily_demand", {})
    ent_demand = obs.get("enterprise_demand", {})
    in_transit = obs.get("in_transit_shipments", [])
    pending_transfers = obs.get("pending_transfers", [])

    action = _default_action()

    inv_totals = _sum_component_totals(inventory)
    comp_dem_totals = _sum_component_totals(demand)

    device_demand_total = 0
    for wh in WAREHOUSES:
        device_demand_total += int(demand.get(wh, {}).get("device", 0) or 0)

    due_soon = {p: 0 for p in COMPONENTS}
    for ship in in_transit:
        prod = ship.get("product_id")
        if prod in COMPONENTS:
            eta = int(ship.get("eta_day", step + 99) or (step + 99))
            if eta <= step + 2:
                due_soon[prod] += int(ship.get("quantity", 0) or 0)
    for tr in pending_transfers:
        prod = tr.get("product_id")
        if prod in COMPONENTS:
            eta = int(tr.get("arrival_day", step + 99) or (step + 99))
            if eta <= step + 2:
                due_soon[prod] += int(tr.get("quantity", 0) or 0)

    urgency = {}
    for p in COMPONENTS:
        total_need = comp_dem_totals[p] + device_demand_total
        available = inv_totals[p] + due_soon[p]
        urgency[p] = max(0, total_need - available)

    product_id = max(COMPONENTS, key=lambda p: (urgency[p], comp_dem_totals[p]))

    if task == "minor-delay":
        supplier_id = "sup_beta" if product_id == "microchip" else "sup_gamma"
    elif task == "port-strike" and 4 <= step <= 9:
        supplier_id = "sup_beta" if product_id == "microchip" else "sup_gamma"
    elif task == "cascading-failures":
        if product_id == "battery" and 3 <= step <= 7:
            supplier_id = "sup_gamma"
        else:
            supplier_id = "sup_alpha"
    else:
        supplier_id = "sup_alpha"

    if supplier_id == "sup_beta" and product_id != "microchip":
        supplier_id = "sup_alpha"
    if supplier_id == "sup_gamma" and product_id != "battery":
        supplier_id = "sup_alpha"

    cap_map = {"sup_alpha": 180, "sup_beta": 120, "sup_gamma": 120}
    cap = cap_map.get(supplier_id, 180)
    if task == "cascading-failures" and step >= 5:
        cap = int(cap * 0.45)

    total_need = comp_dem_totals[product_id] + device_demand_total
    available = inv_totals[product_id] + due_soon[product_id]
    gap = max(0, total_need - available)
    safety = max(15, int(0.30 * max(1, total_need)))
    quantity = max(0, min(cap, gap + safety))

    if task == "port-strike" and 4 <= step <= 9:
        quantity = max(quantity, min(cap, 60))
    if task == "cascading-failures" and step >= 5:
        quantity = max(quantity, min(cap, 70))

    transport = "STANDARD"
    if task == "port-strike" and 4 <= step <= 9 and supplier_id == "sup_alpha":
        transport = "AIR"
    if task == "cascading-failures" and ((3 <= step <= 7 and supplier_id == "sup_alpha") or step >= 5):
        transport = "AIR"

    # Assembly at hub with best component pair availability.
    best_wh = ""
    best_pairs = 0
    for wh in WAREHOUSES:
        pairs = min(
            int(inventory.get(wh, {}).get("microchip", 0) or 0),
            int(inventory.get(wh, {}).get("battery", 0) or 0),
        )
        if pairs > best_pairs:
            best_pairs = pairs
            best_wh = wh

    assemble_qty = 0
    if best_pairs > 0:
        target = int(max(0, device_demand_total) * 0.60)
        if task == "cascading-failures" and step >= 5:
            target = int(max(0, device_demand_total) * 0.90)
        assemble_qty = max(0, min(best_pairs, target))

    # Simple transfer: move urgent component into na_hub when possible.
    transfer_product = ""
    transfer_qty = 0
    transfer_from = ""
    transfer_to = ""
    if task in ("port-strike", "cascading-failures"):
        transfer_product = product_id
        na_need = (
            int(demand.get("na_hub", {}).get(product_id, 0) or 0)
            + int(demand.get("na_hub", {}).get("device", 0) or 0)
        )
        na_have = int(inventory.get("na_hub", {}).get(product_id, 0) or 0)
        need = max(0, na_need - na_have)

        source = ""
        source_surplus = 0
        for wh in ("eu_hub", "apac_hub"):
            wh_have = int(inventory.get(wh, {}).get(product_id, 0) or 0)
            wh_local_need = int(demand.get(wh, {}).get(product_id, 0) or 0)
            surplus = max(0, wh_have - int(0.6 * wh_local_need))
            if surplus > source_surplus:
                source = wh
                source_surplus = surplus

        transfer_qty = min(40, need, source_surplus)
        if transfer_qty >= 8:
            transfer_from = source
            transfer_to = "na_hub"
        else:
            transfer_product = ""
            transfer_qty = 0

    ent_total = 0
    for wh in WAREHOUSES:
        for p in ("microchip", "battery", "device"):
            ent_total += int(ent_demand.get(wh, {}).get(p, 0) or 0)

    if task == "cascading-failures":
        priority = "enterprise"
    elif task == "port-strike" and step >= 5 and ent_total > 0:
        priority = "enterprise"
    else:
        priority = "balanced"

    action.update(
        {
            "supplier_id": supplier_id if quantity > 0 else "",
            "product_id": product_id if quantity > 0 else "",
            "quantity": int(quantity),
            "transport_mode": transport,
            "transfer_product_id": transfer_product,
            "transfer_quantity": int(transfer_qty),
            "transfer_from": transfer_from,
            "transfer_to": transfer_to,
            "assemble_warehouse": best_wh if assemble_qty > 0 else "",
            "assemble_quantity": int(assemble_qty),
            "priority": priority,
        }
    )

    return json.dumps(action)


def get_model_action(
    client: Optional[Any],
    task_name: str,
    step: int,
    obs: Dict[str, Any],
    last_reward: float,
) -> str:
    if client is None:
        return fallback_action(task_name, step, obs)

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
            response_format={"type": "json_object"},
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return fallback_action(task_name, step, obs)


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def run_episode(task_name: str, client: Optional[Any]) -> None:
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
            action_json = get_model_action(client, task_name, step, obs_dict, last_reward)

            try:
                action_dict = json.loads(action_json)
                action_obj = ResilientOpsAction(**action_dict)
                result = await maybe_await(env.step(action_obj))
                reward = float(result.reward.value or 0.0)
                error = None
            except Exception as exc:
                error = str(exc)
                reward = -1.0
                fallback_json = fallback_action(task_name, step, obs_dict)
                action_obj = ResilientOpsAction(**json.loads(fallback_json))
                result = await maybe_await(env.step(action_obj))
                action_json = fallback_json

            done = result.done
            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(
                step=step,
                action=action_json.replace("\n", "").replace("\r", ""),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                score = float(
                    result.info.get("final_score", result.info.get("grader_score", 0.0))
                )
                break

        success = score >= SUCCESS_THRESHOLD

    finally:
        try:
            await maybe_await(env.close())
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client: Optional[Any] = None
    if API_KEY and OpenAI is not None:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception as exc:
            print(f"[DEBUG] OpenAI client init failed: {exc}", flush=True)
            print("[DEBUG] Falling back to deterministic policy", flush=True)
    elif API_KEY and OpenAI is None:
        print("[DEBUG] openai package not installed; falling back to deterministic policy", flush=True)
    else:
        print("[DEBUG] Missing API key; falling back to deterministic policy", flush=True)

    for task_name in resolve_tasks():
        await run_episode(task_name=task_name, client=client)


if __name__ == "__main__":
    asyncio.run(main())