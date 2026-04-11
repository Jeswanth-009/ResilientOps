import asyncio
from env import ResilientOpsEnv, ResilientOpsAction

TASKS = ("minor-delay", "port-strike", "cascading-failures")


async def run_smoke_episode(task_id: str) -> None:
    env = ResilientOpsEnv()
    result = await env.reset(task_id=task_id)

    print(f"\n--- Task: {task_id} ---")
    print(f"Start Day: {result.observation.current_day}")

    for day in range(1, 15):
        if result.done:
            break

        product_id = "microchip" if day % 2 == 1 else "battery"
        transport_mode = "AIR" if task_id == "cascading-failures" and day >= 5 else "STANDARD"

        action = ResilientOpsAction(
            supplier_id="sup_alpha",
            product_id=product_id,
            quantity=120,
            transport_mode=transport_mode,
            assemble_warehouse="na_hub",
            assemble_quantity=12,
            priority="enterprise" if task_id != "minor-delay" else "balanced",
        )
        result = await env.step(action)

    print(f"End Day: {result.observation.current_day}")
    print(f"Done?: {result.done}")
    print(f"Final Score: {float(result.info.get('final_score', 0.0)):.3f}")
    print(f"Final Reward: {result.reward.value:.3f}")

    env.close()


async def test_manual() -> None:
    print("Initializing Environment Smoke Test...")
    for task_id in TASKS:
        await run_smoke_episode(task_id)

if __name__ == "__main__":
    asyncio.run(test_manual())
