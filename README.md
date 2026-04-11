---
title: ResilientOps
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
    - openenv
    - reinforcement-learning
    - supply-chain
license: mit
---

# ResilientOps v2

ResilientOps v2 is a deterministic OpenEnv supply chain operations environment.

It models a 14-day crisis planning horizon where the agent can:
1. Place supplier orders for components.
2. Transfer inventory between warehouses.
3. Assemble devices from components.
4. Prioritize enterprise vs retail demand fulfillment.

## Core Features

- Three warehouses: na_hub, eu_hub, apac_hub.
- Three suppliers with different product coverage and capacity.
- Bill of materials: 1 microchip + 1 battery -> 1 device.
- Customer segmentation with higher enterprise stockout penalties.
- Deterministic inbox messages with task-specific disruption intel.
- Dense economic reward and deterministic task graders.

## Tasks

1. minor-delay (easy)
- sup_alpha lead times are delayed by +2 days.
- objective emphasizes fill rate with shipping-cost discipline.

2. port-strike (medium)
- standard shipping from sup_alpha blocked during the disruption window.
- demand surge requires selective air freight and transfer usage.

3. cascading-failures (hard)
- overlapping disruptions: strike window, supplier capacity cuts, demand spike,
  and sup_beta quality degradation for microchips.
- objective rewards profitability, enterprise SLA protection, fill rate,
  assembly effectiveness, and cost management.

## API

The environment exposes async methods:

- await env.reset(task_id=..., seed=...)
- await env.step(action)
- env.state (property)

## Action Schema

ResilientOpsAction supports compound decisions in one step:

- order fields: supplier_id, product_id, quantity, transport_mode
- transfer fields: transfer_product_id, transfer_quantity, transfer_from, transfer_to
- assembly fields: assemble_warehouse, assemble_quantity
- service priority: priority = enterprise|retail|balanced

## Reward Structure

Per-step raw reward is computed from:

- revenue
- holding_cost
- shipping_cost
- stockout_penalty
- transfer_cost
- assembly_cost

Normalized reward is clamped to [-1.0, 1.0].

## Files

- env.py: environment logic, models, and deterministic graders
- inference.py: baseline LLM-or-fallback inference runner
- openenv.yaml: OpenEnv manifest
- server/app.py: HTTP app entrypoint
- Dockerfile: containerized runtime for deployment

## Local Run

Run baseline inference:

```bash
python inference.py
```

Run smoke test:

```bash
python test_local.py
```

Build and run container:

```bash
docker build -t resilientops:latest .
docker run --rm -p 7860:7860 resilientops:latest
```
