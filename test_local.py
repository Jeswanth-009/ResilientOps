import asyncio
from env import ResilientOpsEnv, ResilientOpsAction

async def test_manual():
    print("Initializing Environment...")
    env = ResilientOpsEnv()
    
    print("\n--- Testing Reset ---")
    result = await env.reset()
    print(f"Start Day: {result.observation.current_day}")
    
    print("\n--- Testing Step ---")
    # Send a dummy action (adjust product/supplier IDs to match what the AI generated)
    dummy_action = ResilientOpsAction(
        supplier_id="SUP_1", 
        product_id="PROD_1", 
        quantity=100, 
        transport_mode="STANDARD"
    )
    
    result = await env.step(dummy_action)
    print(f"Next Day: {result.observation.current_day}")
    print(f"Reward: {result.reward}")
    print(f"Is Done?: {result.done}")
    
    await env.close()

if __name__ == "__main__":
    asyncio.run(test_manual())