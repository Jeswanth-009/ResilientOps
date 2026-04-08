import asyncio
import os
import textwrap
import json
from typing import List, Optional
from openai import OpenAI

# 1. IMPORT YOUR ACTUAL ENVIRONMENT HERE
from env import ResilientOpsAction, ResilientOpsEnv

IMAGE_NAME = os.getenv("IMAGE_NAME") 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini") # Use a fast/cheap model for baseline

TASK_NAME = os.getenv("MY_ENV_V4_TASK", "minor-delay") # Change this to test other tasks
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

def build_user_prompt(step: int, obs: dict, last_reward: float) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}
        Current State: {json.dumps(obs)}
        Last reward: {last_reward:.2f}
        Provide your next action as JSON.
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, obs: dict, last_reward: float) -> str:
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
        # Fallback action if API fails
        return '{"supplier_id": "sup_alpha", "product_id": "microchip", "quantity": 0, "transport_mode": "STANDARD"}'

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # 2. INITIALIZE YOUR SPECIFIC ENVIRONMENT
    env = ResilientOpsEnv() 
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        result = await env.reset()
        last_reward = 0.0
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
                
            obs_dict = result.observation.model_dump()
            action_json_str = get_model_action(client, step, obs_dict, last_reward)
            
            try:
                action_dict = json.loads(action_json_str)
                # 3. USE YOUR ACTION MODEL
                action_obj = ResilientOpsAction(**action_dict)
                result = await env.step(action_obj)
                reward = result.reward.value or 0.0 # Pulling the normalized clamped value
                error = None
            except Exception as e:
                reward = -1.0 
                error = str(e)
                # Apply a blank step to keep time moving
                result = await env.step(ResilientOpsAction(supplier_id="sup_alpha", product_id="microchip", quantity=0, transport_mode="STANDARD"))
            
            done = result.done
            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            
            action_log = action_json_str.replace('\n', '').replace('\r', '')
            log_step(step=step, action=action_log, reward=reward, done=done, error=error)
            
            if done:
                # Capture the final grader score from the info dictionary
                score = result.info.get("final_score", 0.0)
                break
                
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
            
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())