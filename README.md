---
title: ResilientOps
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# ResilientOps: Global Supply Chain Crisis Simulator

This is a custom RL environment built for the OpenEnv competition. 

## Setup
This environment runs natively in a Docker container. 
It requires the `openenv-core` and `pydantic` packages.

## Tasks
1. **Minor Delay (Easy)** - Primary supplier delayed by 2 days.
2. **Port Strike (Medium)** - Standard shipping lane blocked.
3. **Network Collapse (Hard)** - Simultaneous demand spike and raw material shortage.

# ResilientOps

ResilientOps is a deterministic, production-style OpenEnv environment for reinforcement learning in global supply chain crisis management.

It simulates a 14-day planning horizon with:
- Multi-warehouse inventory
- In-transit shipments with lead times
- Daily demand realization
- Task-specific disruption events
- Dense per-day reward signals
- Deterministic, rubric-aligned final grading

## What Is Included

- `env.py`: Full environment implementation and all typed Pydantic models.
- `openenv.yaml`: Environment metadata manifest.
- `Dockerfile`: Python 3.11 container with required dependencies and HF Spaces port 7860.

## Core API

The environment exposes async methods:

- `await env.reset(task_id=..., seed=...)`
- `await env.step(action)`
- `await env.state()`

Primary classes:
- `ResilientOpsEnv`
- `ResilientOpsAction`
- `ResilientOpsObservation`
- `ResilientOpsReward`
- `ResilientOpsStepResult`
- `ResilientOpsState`

## Typed Action Space

`ResilientOpsAction`

- `supplier_id: str`
- `product_id: str`
- `quantity: int`
- `transport_mode: Literal["STANDARD", "AIR"]`

## Typed Observation Space

`ResilientOpsObservation`

- `current_day: int`
- `warehouse_inventory: Dict[str, Dict[str, int]]`
- `in_transit_shipments: List[TransitShipment]`
- `active_crisis_alerts: List[str]`
- `daily_demand: Dict[str, Dict[str, int]]`

## Typed Reward Model

`ResilientOpsReward`

- `value: float` (strictly clamped to [-1.0, 1.0])
- `raw_value: float`
- `revenue: float`
- `holding_cost: float`
- `shipping_cost: float`
- `stockout_penalty: float`

## Daily Reward Function

Every `step()` produces dense feedback (non-sparse):

`reward_raw = revenue - holding_cost - shipping_cost - stockout_penalty`

Where:
- `revenue = fulfilled_units * 30`
- `holding_cost = inventory_units * 0.50`
- `shipping_cost = (AIR units * 10) + (STANDARD units * 1)`
- `stockout_penalty = unmet_units * 20`

The normalized reward used for RL is:

`reward = clip(reward_raw / 5000, -1.0, 1.0)`

## Tasks and Deterministic Graders

### Task 1 (Easy): `minor-delay`

Crisis trigger:
- Primary supplier (`sup_alpha`) has a fixed +2 day delay.

Grader:
- `score = 1.0 - 0.2 * stockout_days`
- Strictly clamped to `[0.0, 1.0]`

### Task 2 (Medium): `port-strike`

Crisis trigger:
- Standard shipping lane blocked for `sup_alpha` during days 4-9.
- Air freight remains available and expensive.

Grader (balanced objective):
- Fill-rate component rewards stockout prevention.
- Cost-efficiency component penalizes excessive transport spend.
- `score = 0.75 * fill_rate + 0.25 * cost_efficiency`
- Strictly clamped to `[0.0, 1.0]`

### Task 3 (Hard): `network-collapse`

Crisis trigger:
- Demand spike starts day 5 (+65%).
- Raw material shortage starts day 5 (supplier capacity -55%).

Grader:
- Normalize agent net profit against deterministic baseline and optimistic bounds:
- `score = (agent_profit - baseline_profit) / (oracle_profit - baseline_profit)`
- Strictly clamped to `[0.0, 1.0]`

## Determinism and Reproducibility

- No random sampling is used in transitions.
- Demand follows a fixed 14-day profile.
- Optional `seed` only shifts deterministic demand profile phase.
- State is fully in-memory and serializable via `await env.state()`.

## Quick Usage Example

```python
import asyncio
from env import ResilientOpsEnv, ResilientOpsAction

async def run_episode():
    env = ResilientOpsEnv()

    result = await env.reset(task_id="port-strike", seed=7)
    done = result.done

    while not done:
        action = ResilientOpsAction(
            supplier_id="sup_alpha",
            product_id="microchip",
            quantity=40,
            transport_mode="AIR",
        )
        result = await env.step(action)
        done = result.done

    final_state = await env.state()
    print("Final grade:", final_state.grader_score)

asyncio.run(run_episode())
```

## Local Docker Setup

Build image:

```bash
docker build -t resilientops:latest .
```

Run container:

```bash
docker run --rm -p 7860:7860 resilientops:latest
```

The container exposes port `7860` for Hugging Face Spaces compatibility.

## Import Contract for Judge Inference Script

`env.py` exports:
- `ResilientOpsEnv`
- `ResilientOpsAction`

So judge-side code can import directly:

```python
from env import ResilientOpsEnv, ResilientOpsAction
```
