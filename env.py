"""
ResilientOps v2: Global Supply Chain Operations Center
======================================================

A deterministic simulation of multi-warehouse electronics supply chain
management during crisis scenarios.

The agent acts as a supply chain operations manager who must simultaneously:

1. PLACE ORDERS with suppliers (choosing supplier, product, quantity, transport mode)
2. TRANSFER INVENTORY between warehouses (responding to regional imbalances)
3. ASSEMBLE DEVICES from components (leveraging bill-of-materials for higher margin)
4. PRIORITIZE CUSTOMER SEGMENTS (enterprise SLA compliance vs retail flexibility)

Key Mechanics:
  - Bill of Materials: 1 microchip + 1 battery → 1 device
    Raw component revenue: $30 + $25 = $55
    Assembled device revenue: $85 − $5 assembly = $80 net
  - Customer Segments: Enterprise (40%, SLA penalty $50/unit) vs Retail (60%, $20/unit)
  - Warehouse-to-warehouse inventory transfers (1-day transit, $1/unit)
  - 4 decision axes per step create combinatorial strategic depth
  - NLP inbox messages requiring comprehension (with noise in hard task)

Determinism: All transitions are fully deterministic. The seed parameter only
shifts the demand multiplier profile index. No random sampling is used.

OpenEnv Compliance:
  - Implements reset()/step()/state() async API
  - Typed Pydantic models for all inputs/outputs
  - Deterministic grader scores strictly in (0.0, 1.0)
  - 14-day episodes with dense per-step reward signal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from openenv.core.env_server.interfaces import Environment as OpenEnvEnvironment
except Exception:
    class OpenEnvEnvironment:
        pass

# ---------------------------------------------------------------------------
# Global simulation constants
# ---------------------------------------------------------------------------

MAX_EPISODE_DAYS: int = 14

PRODUCTS: Tuple[str, ...] = ("microchip", "battery", "device")
COMPONENTS: Tuple[str, ...] = ("microchip", "battery")
WAREHOUSES: Tuple[str, ...] = ("na_hub", "eu_hub", "apac_hub")

# Bill of materials: what components are consumed to assemble one device
BOM: Dict[str, int] = {"microchip": 1, "battery": 1}
ASSEMBLY_COST_PER_UNIT: float = 5.0

# Selling prices ($/unit)
SELL_PRICES: Dict[str, float] = {"microchip": 30.0, "battery": 25.0, "device": 85.0}

# Cost parameters
HOLDING_COST_PER_UNIT: float = 0.50
STANDARD_SHIPPING_PER_UNIT: float = 1.5
AIR_SHIPPING_PER_UNIT: float = 9.0
TRANSFER_COST_PER_UNIT: float = 1.0
RETAIL_STOCKOUT_PENALTY: float = 20.0
ENTERPRISE_STOCKOUT_PENALTY: float = 50.0
ENTERPRISE_FRACTION: float = 0.4
RETAIL_FRACTION: float = 0.6

DAILY_REWARD_SCALE: float = 6000.0
GRADER_EPSILON: float = 0.001

# Deterministic demand multiplier profile (14 days)
BASE_DEMAND_PROFILE: Tuple[float, ...] = (
    1.00, 1.04, 0.96, 1.07, 1.02, 1.10, 0.95,
    1.12, 1.03, 0.99, 1.05, 1.08, 0.97, 1.01,
)

# Inbound allocation fractions by product (deterministic)
ALLOCATION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "microchip": {"na_hub": 0.46, "eu_hub": 0.33, "apac_hub": 0.21},
    "battery": {"na_hub": 0.40, "eu_hub": 0.30, "apac_hub": 0.30},
}

TransportMode = Literal["STANDARD", "AIR"]
PriorityMode = Literal["enterprise", "retail", "balanced"]
TaskId = Literal["minor-delay", "port-strike", "cascading-failures"]


# ---------------------------------------------------------------------------
# Typed public API models
# ---------------------------------------------------------------------------

class ResilientOpsAction(BaseModel):
    """Compound daily action: order + transfer + assemble + prioritize.

    The agent may use any subset of these levers each step. Fields left at
    defaults (empty strings, zero quantities) are simply skipped.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Order placement ---
    supplier_id: str = ""
    product_id: str = ""
    quantity: int = 0
    transport_mode: TransportMode = "STANDARD"

    # --- Warehouse transfer ---
    transfer_product_id: str = ""
    transfer_quantity: int = 0
    transfer_from: str = ""
    transfer_to: str = ""

    # --- Assembly decision ---
    assemble_warehouse: str = ""
    assemble_quantity: int = 0

    # --- Customer segment priority ---
    priority: PriorityMode = "balanced"


class TransitShipment(BaseModel):
    """Shipment currently in transit from a supplier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shipment_id: str
    supplier_id: str
    product_id: str
    quantity: int = Field(..., ge=1)
    transport_mode: TransportMode
    dispatch_day: int = Field(..., ge=1)
    eta_day: int = Field(..., ge=1)


class PendingTransfer(BaseModel):
    """Warehouse-to-warehouse transfer in transit (1-day lead time)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transfer_id: str
    product_id: str
    quantity: int = Field(..., ge=1)
    from_warehouse: str
    to_warehouse: str
    dispatch_day: int = Field(..., ge=1)
    arrival_day: int = Field(..., ge=1)


class ResilientOpsObservation(BaseModel):
    """Observation returned to the agent after reset and each step."""

    model_config = ConfigDict(extra="forbid")

    current_day: int = Field(..., ge=0, le=MAX_EPISODE_DAYS)
    warehouse_inventory: Dict[str, Dict[str, int]]
    in_transit_shipments: List[TransitShipment]
    pending_transfers: List[PendingTransfer]
    daily_demand: Dict[str, Dict[str, int]]
    enterprise_demand: Dict[str, Dict[str, int]]
    retail_demand: Dict[str, Dict[str, int]]
    inbox_messages: List[str]
    last_action_feedback: List[str]
    financial_summary: Dict[str, float]
    service_levels: Dict[str, float]
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResilientOpsReward(BaseModel):
    """Dense per-step reward with transparent accounting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(..., ge=-1.0, le=1.0)
    raw_value: float
    revenue: float
    holding_cost: float
    shipping_cost: float
    stockout_penalty: float
    transfer_cost: float
    assembly_cost: float

    @field_validator("value")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(-1.0, min(1.0, float(v)))


class ResilientOpsStepResult(BaseModel):
    """Unified return object for step()."""

    model_config = ConfigDict(extra="forbid")

    observation: ResilientOpsObservation
    reward: ResilientOpsReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResilientOpsState(BaseModel):
    """Full state snapshot for debugging and reproducibility."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_id: TaskId
    step_count: int = Field(..., ge=0, le=MAX_EPISODE_DAYS)
    current_day: int = Field(..., ge=0, le=MAX_EPISODE_DAYS)
    max_episode_days: int = MAX_EPISODE_DAYS
    warehouse_inventory: Dict[str, Dict[str, int]]
    in_transit_shipments: List[TransitShipment]
    pending_transfers: List[PendingTransfer]
    daily_demand: Dict[str, Dict[str, int]]
    enterprise_demand: Dict[str, Dict[str, int]]
    retail_demand: Dict[str, Dict[str, int]]
    inbox_messages: List[str]
    last_action_feedback: List[str]
    cumulative_profit: float
    total_revenue: float
    total_holding_cost: float
    total_shipping_cost: float
    total_stockout_penalty: float
    total_transfer_cost: float
    total_assembly_cost: float
    total_enterprise_unmet: int
    total_retail_unmet: int
    total_enterprise_demand: int
    total_retail_demand: int
    total_demand_units: int
    total_fulfilled_units: int
    stockout_days: int
    devices_assembled: int
    transfer_units_moved: int
    done: bool
    grader_score: float


# ---------------------------------------------------------------------------
# Internal configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SupplierConfig:
    supplier_id: str
    products: Tuple[str, ...]
    standard_lead_days: int
    air_lead_days: int
    daily_capacity: int


@dataclass(frozen=True)
class TaskConfig:
    task_id: TaskId
    difficulty: Literal["easy", "medium", "hard"]
    title: str
    description: str
    initial_inventory: Dict[str, Dict[str, int]]
    base_daily_demand: Dict[str, Dict[str, int]]


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class ResilientOpsEnv(OpenEnvEnvironment):
    """Deterministic global supply-chain operations center with 3 crisis tasks.

    API contract (async):
      - reset(task_id, seed, episode_id) → StepResult
      - step(action) → StepResult
      - state → State snapshot
    """

    def __init__(self, max_episode_days: int = MAX_EPISODE_DAYS, _skip_bounds: bool = False):
        try:
            super().__init__()
        except TypeError:
            pass

        if max_episode_days != MAX_EPISODE_DAYS:
            raise ValueError(f"Episodes must be exactly {MAX_EPISODE_DAYS} days.")
        self.max_episode_days = max_episode_days
        self._skip_bounds = _skip_bounds

        # --- Suppliers ---
        self._suppliers: Dict[str, SupplierConfig] = {
            "sup_alpha": SupplierConfig(
                supplier_id="sup_alpha",
                products=("microchip", "battery"),
                standard_lead_days=2,
                air_lead_days=1,
                daily_capacity=180,
            ),
            "sup_beta": SupplierConfig(
                supplier_id="sup_beta",
                products=("microchip",),
                standard_lead_days=3,
                air_lead_days=1,
                daily_capacity=120,
            ),
            "sup_gamma": SupplierConfig(
                supplier_id="sup_gamma",
                products=("battery",),
                standard_lead_days=3,
                air_lead_days=1,
                daily_capacity=120,
            ),
        }

        # --- Tasks ---
        self._tasks: Dict[TaskId, TaskConfig] = {
            "minor-delay": TaskConfig(
                task_id="minor-delay",
                difficulty="easy",
                title="Supplier Delay",
                description=(
                    "Primary supplier sup_alpha has a +2 day lead time increase. "
                    "Agent must read inbox, realize sup_alpha is slow, and switch to "
                    "sup_beta (microchip) and sup_gamma (battery) to avoid stockouts."
                ),
                initial_inventory={
                    "na_hub": {"microchip": 130, "battery": 115, "device": 40},
                    "eu_hub": {"microchip": 110, "battery": 100, "device": 35},
                    "apac_hub": {"microchip": 100, "battery": 110, "device": 30},
                },
                base_daily_demand={
                    "na_hub": {"microchip": 10, "battery": 8, "device": 14},
                    "eu_hub": {"microchip": 8, "battery": 7, "device": 11},
                    "apac_hub": {"microchip": 7, "battery": 8, "device": 9},
                },
            ),
            "port-strike": TaskConfig(
                task_id="port-strike",
                difficulty="medium",
                title="Port Strike + Demand Surge",
                description=(
                    "Standard shipping from sup_alpha is blocked days 4-9 by a port strike. "
                    "Demand increases 30% starting day 5. Agent must use air freight "
                    "selectively, transfer inventory between warehouses, and assemble devices."
                ),
                initial_inventory={
                    "na_hub": {"microchip": 100, "battery": 90, "device": 25},
                    "eu_hub": {"microchip": 85, "battery": 80, "device": 20},
                    "apac_hub": {"microchip": 80, "battery": 85, "device": 18},
                },
                base_daily_demand={
                    "na_hub": {"microchip": 11, "battery": 9, "device": 16},
                    "eu_hub": {"microchip": 9, "battery": 8, "device": 13},
                    "apac_hub": {"microchip": 8, "battery": 9, "device": 11},
                },
            ),
            "cascading-failures": TaskConfig(
                task_id="cascading-failures",
                difficulty="hard",
                title="Cascading Failures",
                description=(
                    "Multiple simultaneous disruptions: supplier capacity drops 55% after day 4, "
                    "port strike days 3-7, demand spike 65% after day 5, and sup_beta delivers "
                    "only 75% of ordered microchips after day 4 (quality issues). Agent must "
                    "assemble devices, transfer inventory, prioritize enterprise customers, and "
                    "carefully parse inbox messages (some contain noise/misdirection)."
                ),
                initial_inventory={
                    "na_hub": {"microchip": 95, "battery": 85, "device": 20},
                    "eu_hub": {"microchip": 82, "battery": 78, "device": 16},
                    "apac_hub": {"microchip": 78, "battery": 82, "device": 14},
                },
                base_daily_demand={
                    "na_hub": {"microchip": 12, "battery": 10, "device": 17},
                    "eu_hub": {"microchip": 10, "battery": 9, "device": 14},
                    "apac_hub": {"microchip": 9, "battery": 10, "device": 12},
                },
            ),
        }

        # --- Runtime episode state ---
        self._episode_id: str = ""
        self._task_id: TaskId = "minor-delay"
        self._current_day: int = 0
        self._done: bool = False
        self._demand_offset: int = 0
        self._feedback: List[str] = []

        self._inventory: Dict[str, Dict[str, int]] = {}
        self._in_transit: List[TransitShipment] = []
        self._pending_transfers: List[PendingTransfer] = []
        self._daily_demand: Dict[str, Dict[str, int]] = {}
        self._enterprise_demand: Dict[str, Dict[str, int]] = {}
        self._retail_demand: Dict[str, Dict[str, int]] = {}

        self._metrics: Dict[str, float] = {}
        self._task3_baseline: float = 1.0
        self._task3_oracle: float = 2.0

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def reset(
        self,
        task_id: TaskId = "minor-delay",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> ResilientOpsStepResult:
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task_id '{task_id}'.")

        task = self._tasks[task_id]
        self._task_id = task_id
        self._episode_id = episode_id or str(uuid4())
        self._current_day = 0
        self._done = False
        self._demand_offset = 0 if seed is None else int(seed) % len(BASE_DEMAND_PROFILE)
        self._feedback = ["System initialized. Awaiting your first action."]

        # Deep-copy initial inventory
        self._inventory = {
            wh: {prod: int(qty) for prod, qty in prod_map.items()}
            for wh, prod_map in task.initial_inventory.items()
        }
        # Ensure all warehouses have all products
        for wh in WAREHOUSES:
            for prod in PRODUCTS:
                self._inventory[wh].setdefault(prod, 0)

        self._in_transit = []
        self._pending_transfers = []

        # Compute day-1 demand for the initial observation
        self._compute_demand(day=1)

        self._metrics = {
            "cumulative_profit": 0.0,
            "total_revenue": 0.0,
            "total_holding_cost": 0.0,
            "total_shipping_cost": 0.0,
            "total_stockout_penalty": 0.0,
            "total_transfer_cost": 0.0,
            "total_assembly_cost": 0.0,
            "total_enterprise_unmet": 0.0,
            "total_retail_unmet": 0.0,
            "total_enterprise_demand": 0.0,
            "total_retail_demand": 0.0,
            "total_demand_units": 0.0,
            "total_fulfilled_units": 0.0,
            "stockout_days": 0.0,
            "devices_assembled": 0.0,
            "transfer_units_moved": 0.0,
        }

        if self._task_id == "cascading-failures" and not self._skip_bounds:
            self._task3_baseline, self._task3_oracle = await self._compute_task3_bounds()
        else:
            self._task3_baseline, self._task3_oracle = 1.0, 2.0

        obs = self._build_observation(0)
        reward = ResilientOpsReward(
            value=0.0, raw_value=0.0, revenue=0.0, holding_cost=0.0,
            shipping_cost=0.0, stockout_penalty=0.0, transfer_cost=0.0,
            assembly_cost=0.0,
        )
        return ResilientOpsStepResult(
            observation=obs, reward=reward, done=False,
            info={"task_id": self._task_id, "difficulty": task.difficulty,
                  "crisis_description": task.description},
        )

    async def reset_async(self, seed: Optional[int] = None,
                          episode_id: Optional[str] = None,
                          task_id: TaskId = "minor-delay", **_: Any
                          ) -> ResilientOpsObservation:
        result = await self.reset(task_id=task_id, seed=seed, episode_id=episode_id)
        return self._obs_from_result(result)

    async def step(self, action: ResilientOpsAction) -> ResilientOpsStepResult:
        if not self._episode_id:
            raise RuntimeError("Call reset() first.")
        if self._done:
            obs = self._build_observation(self._current_day)
            zero_r = ResilientOpsReward(value=0.0, raw_value=0.0, revenue=0.0,
                                        holding_cost=0.0, shipping_cost=0.0,
                                        stockout_penalty=0.0, transfer_cost=0.0,
                                        assembly_cost=0.0)
            return ResilientOpsStepResult(observation=obs, reward=zero_r,
                                          done=True, info={"message": "Episode ended."})

        day = self._current_day + 1
        self._feedback = []

        # 1) Receive arriving shipments and transfers
        self._receive_arrivals(day)

        # 2) Process order
        shipping_cost = self._process_order(action, day)

        # 3) Process transfer
        transfer_cost = self._process_transfer(action, day)

        # 4) Process assembly
        assembly_cost = self._process_assembly(action)

        # 5) Compute demand and fulfill
        self._compute_demand(day)
        rev, ent_unmet, ret_unmet, fulfilled = self._fulfill_demand(action.priority)

        # 6) Compute economics
        holding = self._total_inventory_units() * HOLDING_COST_PER_UNIT
        stockout_pen = (ent_unmet * ENTERPRISE_STOCKOUT_PENALTY
                        + ret_unmet * RETAIL_STOCKOUT_PENALTY)
        raw = rev - holding - shipping_cost - stockout_pen - transfer_cost - assembly_cost
        norm = max(-1.0, min(1.0, raw / DAILY_REWARD_SCALE))

        # 7) Update metrics
        self._metrics["total_revenue"] += rev
        self._metrics["total_holding_cost"] += holding
        self._metrics["total_shipping_cost"] += shipping_cost
        self._metrics["total_stockout_penalty"] += stockout_pen
        self._metrics["total_transfer_cost"] += transfer_cost
        self._metrics["total_assembly_cost"] += assembly_cost
        self._metrics["cumulative_profit"] += raw
        self._metrics["total_enterprise_unmet"] += float(ent_unmet)
        self._metrics["total_retail_unmet"] += float(ret_unmet)
        total_ent_d = sum(
            self._enterprise_demand[wh].get(p, 0)
            for wh in WAREHOUSES for p in PRODUCTS
        )
        total_ret_d = sum(
            self._retail_demand[wh].get(p, 0)
            for wh in WAREHOUSES for p in PRODUCTS
        )
        self._metrics["total_enterprise_demand"] += float(total_ent_d)
        self._metrics["total_retail_demand"] += float(total_ret_d)
        self._metrics["total_demand_units"] += float(fulfilled + ent_unmet + ret_unmet)
        self._metrics["total_fulfilled_units"] += float(fulfilled)
        if (ent_unmet + ret_unmet) > 0:
            self._metrics["stockout_days"] += 1.0

        # 8) Advance day
        self._current_day = day
        self._done = day >= self.max_episode_days

        obs = self._build_observation(day)
        reward = ResilientOpsReward(
            value=norm, raw_value=raw, revenue=rev, holding_cost=holding,
            shipping_cost=shipping_cost, stockout_penalty=stockout_pen,
            transfer_cost=transfer_cost, assembly_cost=assembly_cost,
        )
        info: Dict[str, Any] = {
            "task_id": self._task_id, "day": day,
            "feedback": list(self._feedback),
            "fulfilled": fulfilled,
            "enterprise_unmet": ent_unmet,
            "retail_unmet": ret_unmet,
            "grader_score": self.grade(),
        }
        if self._done:
            info["final_score"] = self.grade()

        return ResilientOpsStepResult(observation=obs, reward=reward,
                                      done=self._done, info=info)

    async def step_async(self, action: ResilientOpsAction, **_: Any
                         ) -> ResilientOpsObservation:
        result = await self.step(action)
        return self._obs_from_result(result)

    @property
    def state(self) -> ResilientOpsState:
        if not self._episode_id:
            raise RuntimeError("Call reset() first.")
        return ResilientOpsState(
            episode_id=self._episode_id,
            task_id=self._task_id,
            step_count=self._current_day,
            current_day=self._current_day,
            warehouse_inventory=self._copy_inventory(),
            in_transit_shipments=list(self._in_transit),
            pending_transfers=list(self._pending_transfers),
            daily_demand=self._copy_dict2(self._daily_demand),
            enterprise_demand=self._copy_dict2(self._enterprise_demand),
            retail_demand=self._copy_dict2(self._retail_demand),
            inbox_messages=self._inbox(self._current_day),
            last_action_feedback=list(self._feedback),
            cumulative_profit=self._metrics["cumulative_profit"],
            total_revenue=self._metrics["total_revenue"],
            total_holding_cost=self._metrics["total_holding_cost"],
            total_shipping_cost=self._metrics["total_shipping_cost"],
            total_stockout_penalty=self._metrics["total_stockout_penalty"],
            total_transfer_cost=self._metrics["total_transfer_cost"],
            total_assembly_cost=self._metrics["total_assembly_cost"],
            total_enterprise_unmet=int(self._metrics["total_enterprise_unmet"]),
            total_retail_unmet=int(self._metrics["total_retail_unmet"]),
            total_enterprise_demand=int(self._metrics["total_enterprise_demand"]),
            total_retail_demand=int(self._metrics["total_retail_demand"]),
            total_demand_units=int(self._metrics["total_demand_units"]),
            total_fulfilled_units=int(self._metrics["total_fulfilled_units"]),
            stockout_days=int(self._metrics["stockout_days"]),
            devices_assembled=int(self._metrics["devices_assembled"]),
            transfer_units_moved=int(self._metrics["transfer_units_moved"]),
            done=self._done,
            grader_score=self.grade(),
        )

    async def state_async(self) -> ResilientOpsState:
        return self.state

    # ------------------------------------------------------------------
    # Graders (deterministic, multi-dimensional)
    # ------------------------------------------------------------------

    def grade(self) -> float:
        if self._task_id == "minor-delay":
            return self._grade_minor_delay()
        if self._task_id == "port-strike":
            return self._grade_port_strike()
        return self._grade_cascading()

    def _grade_minor_delay(self) -> float:
        """Easy task: 60% fill rate + 40% cost efficiency."""
        total_d = max(1.0, self._metrics["total_demand_units"])
        total_f = self._metrics["total_fulfilled_units"]
        fill_rate = min(1.0, total_f / total_d)

        ship = self._metrics["total_shipping_cost"]
        cost_cap = total_d * 2.5
        cost_eff = 1.0 - min(1.0, ship / max(1.0, cost_cap))

        score = 0.60 * fill_rate + 0.40 * cost_eff
        return self._clamp01(score)

    def _grade_port_strike(self) -> float:
        """Medium task: 35% fill rate + 30% enterprise SLA + 20% cost + 15% transfer usage."""
        total_d = max(1.0, self._metrics["total_demand_units"])
        total_f = self._metrics["total_fulfilled_units"]
        fill_rate = min(1.0, total_f / total_d)

        ent_d = max(1.0, self._metrics["total_enterprise_demand"])
        ent_u = self._metrics["total_enterprise_unmet"]
        ent_sla = 1.0 - min(1.0, ent_u / ent_d)

        ship = self._metrics["total_shipping_cost"]
        cost_cap = total_d * 4.0
        cost_eff = 1.0 - min(1.0, ship / max(1.0, cost_cap))

        transfer_used = min(1.0, self._metrics["transfer_units_moved"] / max(1.0, total_d * 0.15))

        score = (0.35 * fill_rate + 0.30 * ent_sla + 0.20 * cost_eff
                 + 0.15 * transfer_used)
        return self._clamp01(score)

    def _grade_cascading(self) -> float:
        """Hard task: 35% profit vs oracle + 25% enterprise SLA + 20% fill rate
        + 10% assembly bonus + 10% cost management."""
        agent_profit = self._metrics["cumulative_profit"]
        baseline = self._task3_baseline
        oracle = self._task3_oracle
        denom = max(1.0, oracle - baseline)
        profit_score = max(0.0, min(1.0, (agent_profit - baseline) / denom))

        ent_d = max(1.0, self._metrics["total_enterprise_demand"])
        ent_u = self._metrics["total_enterprise_unmet"]
        ent_sla = 1.0 - min(1.0, ent_u / ent_d)

        total_d = max(1.0, self._metrics["total_demand_units"])
        total_f = self._metrics["total_fulfilled_units"]
        fill_rate = min(1.0, total_f / total_d)

        # Assembly bonus: assembling devices is the high-margin strategy
        assembled = self._metrics["devices_assembled"]
        target_assembly = total_d * 0.25
        assembly_score = min(1.0, assembled / max(1.0, target_assembly))

        # Cost management: keep shipping+transfer reasonable relative to revenue
        total_cost = (self._metrics["total_shipping_cost"]
                      + self._metrics["total_transfer_cost"])
        total_rev = max(1.0, self._metrics["total_revenue"])
        cost_ratio = total_cost / total_rev
        cost_mgmt = 1.0 - min(1.0, cost_ratio / 0.25)

        score = (0.35 * profit_score + 0.25 * ent_sla + 0.20 * fill_rate
                 + 0.10 * assembly_score + 0.10 * cost_mgmt)
        return self._clamp01(score)

    # ------------------------------------------------------------------
    # Order / transfer / assembly processing
    # ------------------------------------------------------------------

    def _process_order(self, action: ResilientOpsAction, day: int) -> float:
        """Validate and dispatch an order. Returns shipping cost."""
        if not action.supplier_id or action.quantity <= 0 or not action.product_id:
            if action.supplier_id or action.quantity > 0:
                self._feedback.append("ORDER: Skipped — missing required fields.")
            return 0.0

        supplier = self._suppliers.get(action.supplier_id)
        if supplier is None:
            self._feedback.append(
                f"ORDER ERROR: Supplier '{action.supplier_id}' does not exist."
            )
            return 0.0

        if action.product_id not in COMPONENTS:
            self._feedback.append(
                f"ORDER ERROR: Can only order components ({', '.join(COMPONENTS)}), not '{action.product_id}'."
            )
            return 0.0

        if action.product_id not in supplier.products:
            self._feedback.append(
                f"ORDER ERROR: {supplier.supplier_id} does not supply '{action.product_id}'."
            )
            return 0.0

        cap = self._effective_capacity(supplier.supplier_id, day)
        if action.quantity > cap:
            self._feedback.append(
                f"ORDER REJECTED: Requested {action.quantity} but {supplier.supplier_id} "
                f"capacity is {cap} units today. Try a smaller order or alternate supplier."
            )
            return 0.0

        # Port-strike blockade
        if (self._task_id == "port-strike"
                and supplier.supplier_id == "sup_alpha"
                and action.transport_mode == "STANDARD"
                and 4 <= day <= 9):
            self._feedback.append(
                "ORDER REJECTED: Standard shipping from sup_alpha is BLOCKED by port "
                "strike (days 4-9)! Switch transport_mode to AIR or use sup_beta/sup_gamma."
            )
            return 0.0

        if (self._task_id == "cascading-failures"
                and supplier.supplier_id == "sup_alpha"
                and action.transport_mode == "STANDARD"
                and 3 <= day <= 7):
            self._feedback.append(
                "ORDER REJECTED: Port strike blocks standard shipping from sup_alpha "
                "(days 3-7). Use AIR or switch suppliers."
            )
            return 0.0

        lead = self._effective_lead_time(supplier, action.transport_mode, day)
        eta = day + lead

        # Quality issue in cascading-failures: sup_beta delivers 75% after day 4
        actual_qty = action.quantity
        if (self._task_id == "cascading-failures"
                and supplier.supplier_id == "sup_beta"
                and day >= 5):
            actual_qty = max(1, int(action.quantity * 0.75))
            self._feedback.append(
                f"ORDER WARNING: sup_beta quality issues — only {actual_qty} of "
                f"{action.quantity} ordered microchips will be delivered."
            )

        ship = TransitShipment(
            shipment_id=str(uuid4()),
            supplier_id=supplier.supplier_id,
            product_id=action.product_id,
            quantity=actual_qty,
            transport_mode=action.transport_mode,
            dispatch_day=day,
            eta_day=eta,
        )
        self._in_transit.append(ship)

        unit_cost = AIR_SHIPPING_PER_UNIT if action.transport_mode == "AIR" else STANDARD_SHIPPING_PER_UNIT
        cost = action.quantity * unit_cost  # Pay for what you ordered, not what arrives

        self._feedback.append(
            f"ORDER OK: {actual_qty} units of {action.product_id} from "
            f"{supplier.supplier_id} via {action.transport_mode}. ETA Day {eta}."
        )
        return cost

    def _process_transfer(self, action: ResilientOpsAction, day: int) -> float:
        """Validate and dispatch a warehouse transfer. Returns transfer cost."""
        if (not action.transfer_product_id or action.transfer_quantity <= 0
                or not action.transfer_from or not action.transfer_to):
            return 0.0

        if action.transfer_from not in WAREHOUSES or action.transfer_to not in WAREHOUSES:
            self._feedback.append(
                f"TRANSFER ERROR: Invalid warehouse(s). Use: {', '.join(WAREHOUSES)}."
            )
            return 0.0

        if action.transfer_from == action.transfer_to:
            self._feedback.append("TRANSFER ERROR: Source and destination must differ.")
            return 0.0

        if action.transfer_product_id not in COMPONENTS:
            self._feedback.append(
                f"TRANSFER ERROR: Can only transfer components, not '{action.transfer_product_id}'."
            )
            return 0.0

        available = self._inventory[action.transfer_from].get(action.transfer_product_id, 0)
        qty = min(action.transfer_quantity, available)
        if qty <= 0:
            self._feedback.append(
                f"TRANSFER SKIPPED: No {action.transfer_product_id} available at "
                f"{action.transfer_from} to transfer."
            )
            return 0.0

        # Deduct from source immediately
        self._inventory[action.transfer_from][action.transfer_product_id] -= qty

        transfer = PendingTransfer(
            transfer_id=str(uuid4()),
            product_id=action.transfer_product_id,
            quantity=qty,
            from_warehouse=action.transfer_from,
            to_warehouse=action.transfer_to,
            dispatch_day=day,
            arrival_day=day + 1,  # 1-day transfer
        )
        self._pending_transfers.append(transfer)
        self._metrics["transfer_units_moved"] += float(qty)

        cost = qty * TRANSFER_COST_PER_UNIT
        self._feedback.append(
            f"TRANSFER OK: {qty} {action.transfer_product_id} from "
            f"{action.transfer_from} → {action.transfer_to}. Arrives Day {day + 1}."
        )
        return cost

    def _process_assembly(self, action: ResilientOpsAction) -> float:
        """Assemble devices from components. Returns assembly cost."""
        if not action.assemble_warehouse or action.assemble_quantity <= 0:
            return 0.0

        if action.assemble_warehouse not in WAREHOUSES:
            self._feedback.append(
                f"ASSEMBLY ERROR: Unknown warehouse '{action.assemble_warehouse}'."
            )
            return 0.0

        wh = action.assemble_warehouse
        requested = action.assemble_quantity

        # Check component availability
        max_by_microchip = self._inventory[wh].get("microchip", 0) // BOM["microchip"]
        max_by_battery = self._inventory[wh].get("battery", 0) // BOM["battery"]
        max_possible = min(max_by_microchip, max_by_battery)

        actual = min(requested, max_possible)
        if actual <= 0:
            self._feedback.append(
                f"ASSEMBLY SKIPPED: Not enough components at {wh} "
                f"(need 1 microchip + 1 battery per device; have "
                f"{self._inventory[wh].get('microchip', 0)} microchip, "
                f"{self._inventory[wh].get('battery', 0)} battery)."
            )
            return 0.0

        # Consume components, produce devices
        self._inventory[wh]["microchip"] -= actual * BOM["microchip"]
        self._inventory[wh]["battery"] -= actual * BOM["battery"]
        self._inventory[wh]["device"] += actual
        self._metrics["devices_assembled"] += float(actual)

        cost = actual * ASSEMBLY_COST_PER_UNIT
        if actual < requested:
            self._feedback.append(
                f"ASSEMBLY PARTIAL: Assembled {actual}/{requested} devices at {wh} "
                f"(limited by component stock)."
            )
        else:
            self._feedback.append(
                f"ASSEMBLY OK: Assembled {actual} devices at {wh}."
            )
        return cost

    # ------------------------------------------------------------------
    # Demand fulfillment with customer segmentation
    # ------------------------------------------------------------------

    def _fulfill_demand(self, priority: PriorityMode) -> Tuple[float, int, int, int]:
        """Fulfill demand from local warehouse inventory with segment priority.

        Returns (revenue, enterprise_unmet, retail_unmet, total_fulfilled).
        """
        total_revenue = 0.0
        total_ent_unmet = 0
        total_ret_unmet = 0
        total_fulfilled = 0

        for wh in WAREHOUSES:
            for prod in PRODUCTS:
                ent_demand = self._enterprise_demand[wh].get(prod, 0)
                ret_demand = self._retail_demand[wh].get(prod, 0)
                available = self._inventory[wh].get(prod, 0)

                if priority == "enterprise":
                    ent_fulfilled = min(available, ent_demand)
                    remaining = available - ent_fulfilled
                    ret_fulfilled = min(remaining, ret_demand)
                elif priority == "retail":
                    ret_fulfilled = min(available, ret_demand)
                    remaining = available - ret_fulfilled
                    ent_fulfilled = min(remaining, ent_demand)
                else:  # balanced — proportional
                    total_d = ent_demand + ret_demand
                    if total_d <= 0 or available <= 0:
                        ent_fulfilled = 0
                        ret_fulfilled = 0
                    else:
                        ent_share = available * (ent_demand / total_d)
                        ret_share = available * (ret_demand / total_d)
                        ent_fulfilled = min(int(round(ent_share)), ent_demand, available)
                        ret_fulfilled = min(int(round(ret_share)), ret_demand,
                                            available - ent_fulfilled)
                        # Fix rounding edge
                        if ent_fulfilled + ret_fulfilled > available:
                            ret_fulfilled = available - ent_fulfilled

                self._inventory[wh][prod] = available - ent_fulfilled - ret_fulfilled
                total_revenue += ((ent_fulfilled + ret_fulfilled) * SELL_PRICES[prod])
                total_ent_unmet += max(0, ent_demand - ent_fulfilled)
                total_ret_unmet += max(0, ret_demand - ret_fulfilled)
                total_fulfilled += ent_fulfilled + ret_fulfilled

        return total_revenue, total_ent_unmet, total_ret_unmet, total_fulfilled

    # ------------------------------------------------------------------
    # Arrivals
    # ------------------------------------------------------------------

    def _receive_arrivals(self, day: int) -> None:
        # Supplier shipments
        remaining: List[TransitShipment] = []
        for ship in self._in_transit:
            if ship.eta_day <= day:
                self._allocate_inbound(ship.product_id, ship.quantity)
            else:
                remaining.append(ship)
        self._in_transit = remaining

        # Warehouse transfers
        remaining_t: List[PendingTransfer] = []
        for tr in self._pending_transfers:
            if tr.arrival_day <= day:
                self._inventory[tr.to_warehouse][tr.product_id] = (
                    self._inventory[tr.to_warehouse].get(tr.product_id, 0) + tr.quantity
                )
            else:
                remaining_t.append(tr)
        self._pending_transfers = remaining_t

    def _allocate_inbound(self, product_id: str, quantity: int) -> None:
        if quantity <= 0 or product_id not in ALLOCATION_WEIGHTS:
            return
        weights = ALLOCATION_WEIGHTS[product_id]
        distributed = 0
        for wh in WAREHOUSES[:-1]:
            units = int(quantity * weights[wh])
            self._inventory[wh][product_id] += units
            distributed += units
        self._inventory[WAREHOUSES[-1]][product_id] += (quantity - distributed)

    # ------------------------------------------------------------------
    # Demand computation
    # ------------------------------------------------------------------

    def _compute_demand(self, day: int) -> None:
        task = self._tasks[self._task_id]
        idx = (day - 1 + self._demand_offset) % len(BASE_DEMAND_PROFILE)
        mult = BASE_DEMAND_PROFILE[idx]

        # Crisis modifiers
        if self._task_id == "port-strike" and day >= 5:
            mult *= 1.30
        if self._task_id == "cascading-failures" and day >= 5:
            mult *= 1.65

        self._daily_demand = {}
        self._enterprise_demand = {}
        self._retail_demand = {}

        for wh in WAREHOUSES:
            self._daily_demand[wh] = {}
            self._enterprise_demand[wh] = {}
            self._retail_demand[wh] = {}
            for prod in PRODUCTS:
                base = task.base_daily_demand[wh].get(prod, 0)
                total = max(0, int(round(base * mult)))
                ent = int(round(total * ENTERPRISE_FRACTION))
                ret = total - ent
                self._daily_demand[wh][prod] = total
                self._enterprise_demand[wh][prod] = ent
                self._retail_demand[wh][prod] = ret

    # ------------------------------------------------------------------
    # Supplier crisis modifiers
    # ------------------------------------------------------------------

    def _effective_capacity(self, supplier_id: str, day: int) -> int:
        base = self._suppliers[supplier_id].daily_capacity
        if self._task_id == "cascading-failures" and day >= 5:
            return max(0, int(base * 0.45))
        return base

    def _effective_lead_time(self, supplier: SupplierConfig,
                             mode: TransportMode, day: int) -> int:
        lead = supplier.air_lead_days if mode == "AIR" else supplier.standard_lead_days
        if self._task_id == "minor-delay" and supplier.supplier_id == "sup_alpha":
            lead += 2
        if self._task_id == "cascading-failures" and mode == "STANDARD" and day >= 5:
            lead += 1
        return max(1, lead)

    # ------------------------------------------------------------------
    # Inbox messages
    # ------------------------------------------------------------------

    def _inbox(self, day: int) -> List[str]:
        if self._task_id == "minor-delay":
            return [
                "[Logistics] NOTICE: sup_alpha is experiencing a labor shortage. "
                "All their shipments have +2 days added to lead times until further notice.",
                "[Sales] Enterprise customers are expecting on-time delivery this quarter. "
                "Maintaining high fill rates is critical for contract renewal.",
            ]

        if self._task_id == "port-strike":
            msgs = [
                "[Procurement] sup_beta (microchip specialist) and sup_gamma (battery specialist) "
                "are unaffected by the port issues. Consider ramping orders with them.",
            ]
            if day <= 3:
                msgs.append(
                    "[News] Union workers at Western Port threaten walkout starting Day 4. "
                    "Standard surface shipping from sup_alpha may be disrupted."
                )
            elif day <= 9:
                msgs.append(
                    "[URGENT: Logistics] Port strike ACTIVE (Days 4-9). ALL standard shipments "
                    "from sup_alpha are BLOCKED. Use AIR transport or switch to sup_beta/sup_gamma."
                )
            else:
                msgs.append(
                    "[News] Port strike has ended. Standard shipping from sup_alpha is restored."
                )
            if day >= 5:
                msgs.append(
                    "[Sales] Demand surge detected — 30% increase across all hubs. "
                    "Consider warehouse transfers to rebalance inventory."
                )
            return msgs

        # cascading-failures
        msgs: List[str] = []
        if day <= 2:
            msgs.append(
                "[Intel] Rumors of raw material shortages and possible port disruptions. "
                "Build safety stock and pre-position components."
            )
            msgs.append(
                "[Misinformation//UNVERIFIED] A blog post claims sup_beta has doubled "
                "capacity. Treat with skepticism — no official confirmation."
            )
        elif day <= 4:
            msgs.append(
                "[Logistics] Port strike imminent — standard shipping from sup_alpha "
                "will be BLOCKED Days 3-7. Prepare alternative routes."
            )
            msgs.append(
                "[Procurement] Raw material shortages confirmed. All supplier capacities "
                "will drop ~55% after Day 4. Secure air freight immediately."
            )
        elif day <= 7:
            msgs.append(
                "[CRITICAL: Operations] Port strike active (Days 3-7). Capacity crisis "
                "ongoing. sup_alpha standard orders blocked. Use AIR or sup_beta/sup_gamma."
            )
            msgs.append(
                "[Quality] sup_beta is experiencing manufacturing defects. Only ~75% of "
                "ordered microchips pass QC. Factor this into your planning."
            )
            msgs.append(
                "[Sales] Demand has SPIKED 65% across all hubs! Enterprise SLAs at risk. "
                "Assemble devices for maximum margin and prioritize enterprise fulfillment."
            )
            msgs.append(
                "[Misinformation//UNVERIFIED] An anonymous tip claims the strike ends "
                "tomorrow. Do NOT rely on this — official channels say Days 3-7."
            )
        else:
            msgs.append(
                "[Logistics] Port strike has ended. Standard shipping from sup_alpha restored. "
                "Capacity crisis continues — supplier capacities still reduced."
            )
            msgs.append(
                "[Sales] Demand remains elevated. Continue assembling devices and "
                "prioritizing enterprise customers for SLA compliance."
            )
        return msgs

    # ------------------------------------------------------------------
    # Task 3 profit bounds
    # ------------------------------------------------------------------

    async def _compute_task3_bounds(self) -> Tuple[float, float]:
        baseline = await self._rollout_task3("baseline")
        oracle = await self._rollout_task3("oracle")
        if oracle <= baseline:
            oracle = baseline + 1.0
        return float(baseline), float(oracle)

    @staticmethod
    def _task3_action(policy: str, day: int) -> ResilientOpsAction:
        if policy == "baseline":
            return ResilientOpsAction(
                supplier_id="sup_alpha", product_id="microchip",
                quantity=0, transport_mode="STANDARD",
            )
        # Oracle: aggressive ordering + assembly
        prod = "microchip" if day % 2 == 1 else "battery"
        return ResilientOpsAction(
            supplier_id="sup_alpha", product_id=prod,
            quantity=180, transport_mode="AIR",
            assemble_warehouse="na_hub",
            assemble_quantity=50,
            priority="enterprise",
        )

    async def _rollout_task3(self, policy: str) -> float:
        probe = ResilientOpsEnv(_skip_bounds=True)
        result = await probe.reset(task_id="cascading-failures", seed=self._demand_offset)
        day = 0
        while not result.done and day < self.max_episode_days:
            day += 1
            action = self._task3_action(policy, day)
            result = await probe.step(action)
        return probe._metrics["cumulative_profit"]

    # ------------------------------------------------------------------
    # Observation / state helpers
    # ------------------------------------------------------------------

    def _build_observation(self, day: int) -> ResilientOpsObservation:
        ent_d = max(1.0, self._metrics["total_enterprise_demand"])
        ret_d = max(1.0, self._metrics["total_retail_demand"])
        ent_u = self._metrics["total_enterprise_unmet"]
        ret_u = self._metrics["total_retail_unmet"]

        return ResilientOpsObservation(
            current_day=day,
            warehouse_inventory=self._copy_inventory(),
            in_transit_shipments=list(self._in_transit),
            pending_transfers=list(self._pending_transfers),
            daily_demand=self._copy_dict2(self._daily_demand),
            enterprise_demand=self._copy_dict2(self._enterprise_demand),
            retail_demand=self._copy_dict2(self._retail_demand),
            inbox_messages=self._inbox(max(1, day)),
            last_action_feedback=list(self._feedback) if self._feedback
                else ["No action processed yet."],
            financial_summary={
                "cumulative_profit": round(self._metrics["cumulative_profit"], 2),
                "total_revenue": round(self._metrics["total_revenue"], 2),
                "total_shipping_cost": round(self._metrics["total_shipping_cost"], 2),
                "total_stockout_penalty": round(self._metrics["total_stockout_penalty"], 2),
                "total_transfer_cost": round(self._metrics["total_transfer_cost"], 2),
                "total_assembly_cost": round(self._metrics["total_assembly_cost"], 2),
            },
            service_levels={
                "enterprise_fill_rate": round(
                    max(0.0, 1.0 - ent_u / ent_d), 4),
                "retail_fill_rate": round(
                    max(0.0, 1.0 - ret_u / ret_d), 4),
                "overall_fill_rate": round(
                    max(0.0, self._metrics["total_fulfilled_units"]
                        / max(1.0, self._metrics["total_demand_units"])), 4),
            },
        )

    @staticmethod
    def _obs_from_result(result: ResilientOpsStepResult) -> ResilientOpsObservation:
        return result.observation.model_copy(update={
            "done": result.done,
            "reward": float(result.reward.value),
            "metadata": dict(result.info),
        })

    def _copy_inventory(self) -> Dict[str, Dict[str, int]]:
        return {
            wh: {prod: int(val) for prod, val in prod_map.items()}
            for wh, prod_map in self._inventory.items()
        }

    @staticmethod
    def _copy_dict2(d: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        return {
            k: {kk: int(vv) for kk, vv in v.items()}
            for k, v in d.items()
        }

    def _total_inventory_units(self) -> int:
        return sum(
            self._inventory[wh].get(prod, 0)
            for wh in WAREHOUSES for prod in PRODUCTS
        )

    @staticmethod
    def _clamp01(v: float) -> float:
        return max(GRADER_EPSILON, min(1.0 - GRADER_EPSILON, float(v)))

    def close(self) -> None:
        pass


__all__ = [
    "ResilientOpsAction",
    "ResilientOpsObservation",
    "ResilientOpsReward",
    "ResilientOpsStepResult",
    "ResilientOpsState",
    "ResilientOpsEnv",
    "TransitShipment",
    "PendingTransfer",
]