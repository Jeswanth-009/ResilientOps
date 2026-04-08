"""ResilientOps: Global supply chain crisis simulator for OpenEnv-style RL tasks.

This module intentionally keeps all simulation state in memory and deterministic.
The environment exposes async reset/step/state methods and strongly typed Pydantic
models for action, observation, reward, and state snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    # Prefer the official OpenEnv abstract base when present.
    from openenv.core.env_server.interfaces import Environment as OpenEnvEnvironment
except Exception:
    # Fallback base keeps the class importable in lightweight local setups.
    class OpenEnvEnvironment:
        pass

# -----------------------------
# Global simulation parameters
# -----------------------------

MAX_EPISODE_DAYS: int = 14

SELL_PRICE_PER_UNIT: float = 30.0
HOLDING_COST_PER_UNIT: float = 0.50
AIR_SHIPPING_COST_PER_UNIT: float = 10.0
STANDARD_SHIPPING_COST_PER_UNIT: float = 1.0
STOCKOUT_PENALTY_PER_UNIT: float = 20.0

# This scale converts raw dollar profit/loss into a stable RL training signal.
# Any value beyond this range is clipped to [-1, 1].
DAILY_REWARD_NORMALIZATION_SCALE: float = 5000.0
# Phase-2 validators require strict open interval task scores: 0 < score < 1.
GRADER_EPSILON: float = 0.001

WAREHOUSES: Tuple[str, ...] = ("na_hub", "eu_hub", "apac_hub")
PRODUCTS: Tuple[str, ...] = ("microchip", "battery")

# Deterministic day-by-day demand profile multipliers for a 14-day episode.
# No random sampling is used in this environment.
BASE_DEMAND_PROFILE: Tuple[float, ...] = (
    1.00,
    1.04,
    0.96,
    1.07,
    1.02,
    1.10,
    0.95,
    1.12,
    1.03,
    0.99,
    1.05,
    1.08,
    0.97,
    1.01,
)

# Deterministic inbound allocation fractions by product.
# Remaining units after integer rounding are assigned to the last warehouse.
ALLOCATION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "microchip": {"na_hub": 0.46, "eu_hub": 0.33, "apac_hub": 0.21},
    "battery": {"na_hub": 0.40, "eu_hub": 0.30, "apac_hub": 0.30},
}

TaskId = Literal["minor-delay", "port-strike", "network-collapse"]
TransportMode = Literal["STANDARD", "AIR"]


# -----------------------------
# Typed public API models
# -----------------------------

class ResilientOpsAction(BaseModel):
    """Single daily procurement action taken by the agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    supplier_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)
    quantity: int = Field(..., ge=0, le=10_000)
    transport_mode: Literal["STANDARD", "AIR"]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TransitShipment(BaseModel):
    """Shipment currently in transit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shipment_id: str
    supplier_id: str
    product_id: str
    quantity: int = Field(..., ge=1)
    transport_mode: Literal["STANDARD", "AIR"]
    dispatch_day: int = Field(..., ge=1)
    eta_day: int = Field(..., ge=1)


class ResilientOpsObservation(BaseModel):
    """Observation returned to the agent after reset and each step."""

    model_config = ConfigDict(extra="forbid")

    current_day: int = Field(..., ge=0, le=MAX_EPISODE_DAYS)
    warehouse_inventory: Dict[str, Dict[str, int]]
    in_transit_shipments: List[TransitShipment]
    active_crisis_alerts: List[str]
    daily_demand: Dict[str, Dict[str, int]]
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResilientOpsReward(BaseModel):
    """Dense per-step reward with transparent accounting components."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(..., ge=-1.0, le=1.0)
    raw_value: float
    revenue: float
    holding_cost: float
    shipping_cost: float
    stockout_penalty: float

    @field_validator("value")
    @classmethod
    def _validate_clamped_reward(cls, value: float) -> float:
        # Enforce strict reward range for stable training contracts.
        return max(-1.0, min(1.0, float(value)))


class ResilientOpsStepResult(BaseModel):
    """Unified async API return object."""

    model_config = ConfigDict(extra="forbid")

    observation: ResilientOpsObservation
    reward: ResilientOpsReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResilientOpsState(BaseModel):
    """Full in-memory state snapshot for reproducibility and debugging."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_id: TaskId
    step_count: int = Field(..., ge=0, le=MAX_EPISODE_DAYS)
    current_day: int = Field(..., ge=0, le=MAX_EPISODE_DAYS)
    max_episode_days: int = MAX_EPISODE_DAYS
    warehouse_inventory: Dict[str, Dict[str, int]]
    in_transit_shipments: List[TransitShipment]
    daily_demand: Dict[str, Dict[str, int]]
    active_crisis_alerts: List[str]
    cumulative_profit: float
    total_revenue: float
    total_holding_cost: float
    total_shipping_cost: float
    total_stockout_penalty: float
    total_unmet_units: int
    total_demand_units: int
    stockout_days: int
    done: bool
    grader_score: float


# -----------------------------
# Internal deterministic configs
# -----------------------------

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


class ResilientOpsEnv(OpenEnvEnvironment):
    """Deterministic global supply-chain environment with 3 fixed crisis tasks.

    API contract:
    - async reset(...)
    - async step(action)
    - async state()

    All transitions are deterministic. The optional seed controls only a profile
    index offset for demand multipliers, preserving reproducibility.
    """

    def __init__(self, max_episode_days: int = MAX_EPISODE_DAYS):
        try:
            super().__init__()
        except TypeError:
            # Fallback base may not accept init arguments or may be plain object.
            pass

        self.max_episode_days = int(max_episode_days)

        if self.max_episode_days != MAX_EPISODE_DAYS:
            raise ValueError(
                f"ResilientOps episodes must be exactly {MAX_EPISODE_DAYS} days."
            )

        self._suppliers: Dict[str, SupplierConfig] = {
            "sup_alpha": SupplierConfig(
                supplier_id="sup_alpha",
                products=("microchip", "battery"),
                standard_lead_days=2,
                air_lead_days=1,
                daily_capacity=140,
            ),
            "sup_beta": SupplierConfig(
                supplier_id="sup_beta",
                products=("microchip",),
                standard_lead_days=3,
                air_lead_days=1,
                daily_capacity=95,
            ),
            "sup_gamma": SupplierConfig(
                supplier_id="sup_gamma",
                products=("battery",),
                standard_lead_days=3,
                air_lead_days=1,
                daily_capacity=95,
            ),
        }

        self._tasks: Dict[TaskId, TaskConfig] = {
            "minor-delay": TaskConfig(
                task_id="minor-delay",
                difficulty="easy",
                title="Task 1: Minor Delay",
                description="Primary supplier incurs a deterministic 2-day delay.",
                initial_inventory={
                    "na_hub": {"microchip": 125, "battery": 110},
                    "eu_hub": {"microchip": 110, "battery": 105},
                    "apac_hub": {"microchip": 100, "battery": 115},
                },
                base_daily_demand={
                    "na_hub": {"microchip": 22, "battery": 18},
                    "eu_hub": {"microchip": 17, "battery": 15},
                    "apac_hub": {"microchip": 14, "battery": 16},
                },
            ),
            "port-strike": TaskConfig(
                task_id="port-strike",
                difficulty="medium",
                title="Task 2: Port Strike",
                description=(
                    "Major standard shipping lane is blocked. Agent should use "
                    "air freight selectively to avoid stockouts while managing costs."
                ),
                initial_inventory={
                    "na_hub": {"microchip": 95, "battery": 92},
                    "eu_hub": {"microchip": 88, "battery": 86},
                    "apac_hub": {"microchip": 82, "battery": 90},
                },
                base_daily_demand={
                    "na_hub": {"microchip": 24, "battery": 20},
                    "eu_hub": {"microchip": 19, "battery": 17},
                    "apac_hub": {"microchip": 16, "battery": 18},
                },
            ),
            "network-collapse": TaskConfig(
                task_id="network-collapse",
                difficulty="hard",
                title="Task 3: Network Collapse",
                description=(
                    "Simultaneous demand spike and raw material shortage create a "
                    "combinatorial planning problem over a 14-day horizon."
                ),
                initial_inventory={
                    "na_hub": {"microchip": 90, "battery": 85},
                    "eu_hub": {"microchip": 82, "battery": 80},
                    "apac_hub": {"microchip": 78, "battery": 88},
                },
                base_daily_demand={
                    "na_hub": {"microchip": 25, "battery": 21},
                    "eu_hub": {"microchip": 20, "battery": 18},
                    "apac_hub": {"microchip": 17, "battery": 19},
                },
            ),
        }

        # Runtime episode state containers (initialized in reset).
        self._episode_id: str = ""
        self._task_id: TaskId = "minor-delay"
        self._current_day: int = 0
        self._done: bool = False
        self._demand_profile_offset: int = 0

        self._inventory: Dict[str, Dict[str, int]] = {}
        self._in_transit_shipments: List[TransitShipment] = []
        self._daily_demand: Dict[str, Dict[str, int]] = {}

        self._metrics: Dict[str, float] = {}
        self._task3_baseline_profit: float = 1.0
        self._task3_oracle_profit: float = 2.0

    # -----------------------------
    # Public async API
    # -----------------------------

    async def reset(
        self,
        task_id: TaskId = "minor-delay",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> ResilientOpsStepResult:
        """Reset episode state and return an initial observation/result.

        This is deterministic. The optional seed only offsets the fixed demand
        profile index, making trajectories reproducible for benchmarking.
        """

        if task_id not in self._tasks:
            raise ValueError(f"Unknown task_id '{task_id}'.")

        task = self._tasks[task_id]
        self._task_id = task_id
        self._episode_id = episode_id or str(uuid4())
        self._current_day = 0
        self._done = False

        # Seed affects only deterministic profile phase (no random sampling).
        self._demand_profile_offset = 0 if seed is None else int(seed) % len(BASE_DEMAND_PROFILE)

        # Deep-copy nested dicts to preserve clean in-memory state isolation.
        self._inventory = {
            warehouse: {
                product: int(quantity)
                for product, quantity in product_map.items()
            }
            for warehouse, product_map in task.initial_inventory.items()
        }

        self._in_transit_shipments = []
        self._daily_demand = self._compute_daily_demand(day=1)

        # Core deterministic metrics used by all graders.
        self._metrics = {
            "cumulative_profit": 0.0,
            "total_revenue": 0.0,
            "total_holding_cost": 0.0,
            "total_shipping_cost": 0.0,
            "total_stockout_penalty": 0.0,
            "total_unmet_units": 0.0,
            "total_demand_units": 0.0,
            "stockout_days": 0.0,
            "air_units": 0.0,
            "total_ordered_units": 0.0,
        }

        # Precompute hard-task deterministic scoring anchors.
        if self._task_id == "network-collapse":
            self._task3_baseline_profit, self._task3_oracle_profit = self._compute_task3_profit_bounds()
        else:
            self._task3_baseline_profit = 1.0
            self._task3_oracle_profit = 2.0

        observation = self._build_observation(current_day=0)
        reward = ResilientOpsReward(
            value=0.0,
            raw_value=0.0,
            revenue=0.0,
            holding_cost=0.0,
            shipping_cost=0.0,
            stockout_penalty=0.0,
        )

        return ResilientOpsStepResult(
            observation=observation,
            reward=reward,
            done=False,
            info={
                "task_id": self._task_id,
                "difficulty": self._tasks[self._task_id].difficulty,
                "crisis_description": self._tasks[self._task_id].description,
            },
        )

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: TaskId = "minor-delay",
        **_: Any,
    ) -> ResilientOpsObservation:
        """OpenEnv server compatibility: return an Observation-shaped payload."""

        result = await self.reset(task_id=task_id, seed=seed, episode_id=episode_id)
        return self._as_openenv_observation(result)

    async def step(self, action: ResilientOpsAction) -> ResilientOpsStepResult:
        """Advance the simulation by exactly one day.

        Daily reward equation (dense, non-sparse):
            reward_raw = revenue - holding_cost - shipping_cost - stockout_penalty

        Where:
            - holding_cost = 0.50 per remaining inventory unit
            - shipping_cost = 10.0 per AIR unit, 1.0 per STANDARD unit
            - stockout_penalty = 20.0 per unmet demand unit
        """

        if self._episode_id == "":
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._done:
            # Deterministic terminal behavior after episode completion.
            observation = self._build_observation(current_day=self._current_day)
            reward = ResilientOpsReward(
                value=0.0,
                raw_value=0.0,
                revenue=0.0,
                holding_cost=0.0,
                shipping_cost=0.0,
                stockout_penalty=0.0,
            )
            return ResilientOpsStepResult(
                observation=observation,
                reward=reward,
                done=True,
                info={"message": "Episode already completed."},
            )

        sim_day = self._current_day + 1

        # 1) Receive shipments that arrive today.
        self._receive_arrivals(sim_day=sim_day)

        # 2) Attempt to dispatch today's action as a new shipment.
        dispatched_quantity, shipping_cost, dispatch_note = self._dispatch_action(
            action=action,
            sim_day=sim_day,
        )

        # 3) Build deterministic demand for the day and fulfill from local inventory.
        self._daily_demand = self._compute_daily_demand(day=sim_day)
        fulfilled_units, unmet_units = self._fulfill_daily_demand(self._daily_demand)

        # 4) Compute economics using the exact reward equation required.
        revenue = fulfilled_units * SELL_PRICE_PER_UNIT
        holding_cost = self._total_inventory_units() * HOLDING_COST_PER_UNIT
        stockout_penalty = unmet_units * STOCKOUT_PENALTY_PER_UNIT

        raw_reward = revenue - holding_cost - shipping_cost - stockout_penalty
        normalized_reward = self._normalize_daily_reward(raw_reward)

        # 5) Update deterministic cumulative metrics for grading.
        self._metrics["total_revenue"] += revenue
        self._metrics["total_holding_cost"] += holding_cost
        self._metrics["total_shipping_cost"] += shipping_cost
        self._metrics["total_stockout_penalty"] += stockout_penalty
        self._metrics["total_unmet_units"] += float(unmet_units)
        self._metrics["total_demand_units"] += float(fulfilled_units + unmet_units)
        self._metrics["cumulative_profit"] += raw_reward
        self._metrics["total_ordered_units"] += float(dispatched_quantity)
        if action.transport_mode == "AIR":
            self._metrics["air_units"] += float(dispatched_quantity)

        # Task 1 scoring depends on stockout day count, and this statistic is
        # also reused by the other tasks as a useful diagnostic.
        if self._any_warehouse_out_of_stock():
            self._metrics["stockout_days"] += 1.0

        # 6) Commit day progress and check terminal condition.
        self._current_day = sim_day
        self._done = self._current_day >= self.max_episode_days

        observation = self._build_observation(current_day=self._current_day)
        reward = ResilientOpsReward(
            value=normalized_reward,
            raw_value=raw_reward,
            revenue=revenue,
            holding_cost=holding_cost,
            shipping_cost=shipping_cost,
            stockout_penalty=stockout_penalty,
        )

        info: Dict[str, Any] = {
            "task_id": self._task_id,
            "day": sim_day,
            "dispatch_note": dispatch_note,
            "dispatched_quantity": dispatched_quantity,
            "fulfilled_units": fulfilled_units,
            "unmet_units": unmet_units,
            "grader_score": self.grade(),
        }

        # Provide final score in terminal transition for convenience.
        if self._done:
            info["final_score"] = self.grade()

        return ResilientOpsStepResult(
            observation=observation,
            reward=reward,
            done=self._done,
            info=info,
        )

    async def step_async(
        self,
        action: ResilientOpsAction,
        timeout_s: Optional[float] = None,
        **_: Any,
    ) -> ResilientOpsObservation:
        """OpenEnv server compatibility: return an Observation-shaped payload."""

        del timeout_s
        result = await self.step(action)
        return self._as_openenv_observation(result)

    @property
    def state(self) -> ResilientOpsState:
        """Return a full deterministic state snapshot."""

        if self._episode_id == "":
            raise RuntimeError("Environment not initialized. Call reset() first.")

        return ResilientOpsState(
            episode_id=self._episode_id,
            task_id=self._task_id,
            step_count=self._current_day,
            current_day=self._current_day,
            warehouse_inventory=self._copy_inventory(),
            in_transit_shipments=list(self._in_transit_shipments),
            daily_demand=self._copy_demand(),
            active_crisis_alerts=self._active_crisis_alerts_for_day(self._current_day),
            cumulative_profit=float(self._metrics["cumulative_profit"]),
            total_revenue=float(self._metrics["total_revenue"]),
            total_holding_cost=float(self._metrics["total_holding_cost"]),
            total_shipping_cost=float(self._metrics["total_shipping_cost"]),
            total_stockout_penalty=float(self._metrics["total_stockout_penalty"]),
            total_unmet_units=int(self._metrics["total_unmet_units"]),
            total_demand_units=int(self._metrics["total_demand_units"]),
            stockout_days=int(self._metrics["stockout_days"]),
            done=self._done,
            grader_score=self.grade(),
        )

    async def state_async(self) -> ResilientOpsState:
        """Async state accessor for callers that use await semantics."""

        return self.state

    # -----------------------------
    # Task graders (deterministic)
    # -----------------------------

    def grade(self) -> float:
        """Dispatch to task-specific grader and return score in (0.0, 1.0)."""

        if self._task_id == "minor-delay":
            return self._grade_minor_delay()
        if self._task_id == "port-strike":
            return self._grade_port_strike()
        return self._grade_network_collapse()

    def _grade_minor_delay(self) -> float:
        """Task 1 grader.

        Specification:
            score = 1.0 if no warehouse ever hits 0 inventory,
                    minus 0.2 for each day out of stock.
        """

        stockout_days = float(self._metrics["stockout_days"])
        score = 1.0 - (0.2 * stockout_days)
        return self._clamp01(score)

    def _grade_port_strike(self) -> float:
        """Task 2 grader.

        Objective:
        - maximize stock availability (primary objective)
        - minimize transport spend (secondary objective)

        Formula (deterministic and clamped):
            fill_rate = 1 - unmet / total_demand
            cost_efficiency = 1 - min(1, transport_cost / cost_cap)
            score = 0.75 * fill_rate + 0.25 * cost_efficiency
        """

        total_demand = max(1.0, float(self._metrics["total_demand_units"]))
        unmet = float(self._metrics["total_unmet_units"])
        fill_rate = 1.0 - (unmet / total_demand)

        shipping_cost = float(self._metrics["total_shipping_cost"])
        # Cap tuned to represent a "reasonable" mixed-mode transport budget.
        cost_cap = total_demand * 3.5
        cost_efficiency = 1.0 - min(1.0, shipping_cost / max(1.0, cost_cap))

        score = (0.75 * fill_rate) + (0.25 * cost_efficiency)
        return self._clamp01(score)

    def _grade_network_collapse(self) -> float:
        """Task 3 grader.

        Hard-task score uses normalized net profit against deterministic bounds:
            score = (agent_profit - baseline_profit) / (oracle_profit - baseline_profit)

        Output is strictly clamped to (0.0, 1.0).
        """

        agent_profit = float(self._metrics["cumulative_profit"])
        baseline = float(self._task3_baseline_profit)
        oracle = float(self._task3_oracle_profit)

        denominator = max(1.0, oracle - baseline)
        score = (agent_profit - baseline) / denominator
        return self._clamp01(score)

    # -----------------------------
    # Crisis and transition logic
    # -----------------------------

    def _dispatch_action(
        self,
        action: ResilientOpsAction,
        sim_day: int,
    ) -> Tuple[int, float, str]:
        """Validate and dispatch an order shipment for future arrival."""

        supplier = self._suppliers.get(action.supplier_id)
        if supplier is None:
            return 0, 0.0, "unknown_supplier"

        if action.product_id not in PRODUCTS:
            return 0, 0.0, "unknown_product"

        if action.product_id not in supplier.products:
            return 0, 0.0, "supplier_does_not_supply_product"

        if action.quantity <= 0:
            return 0, 0.0, "no_op"

        # Apply dynamic capacity constraints for crisis scenarios.
        effective_capacity = self._effective_supplier_capacity(
            supplier_id=supplier.supplier_id,
            sim_day=sim_day,
        )
        dispatch_qty = min(int(action.quantity), effective_capacity)

        if dispatch_qty <= 0:
            return 0, 0.0, "capacity_exhausted"

        # Port strike behavior: standard lane blocked in Task 2 for sup_alpha.
        if (
            self._task_id == "port-strike"
            and supplier.supplier_id == "sup_alpha"
            and action.transport_mode == "STANDARD"
            and 4 <= sim_day <= 9
        ):
            return 0, 0.0, "port_strike_blocked_standard_lane"

        lead_days = self._effective_lead_time(
            supplier=supplier,
            transport_mode=action.transport_mode,
            sim_day=sim_day,
        )

        eta_day = sim_day + lead_days

        shipment = TransitShipment(
            shipment_id=str(uuid4()),
            supplier_id=supplier.supplier_id,
            product_id=action.product_id,
            quantity=dispatch_qty,
            transport_mode=action.transport_mode,
            dispatch_day=sim_day,
            eta_day=eta_day,
        )
        self._in_transit_shipments.append(shipment)

        unit_shipping_cost = (
            AIR_SHIPPING_COST_PER_UNIT
            if action.transport_mode == "AIR"
            else STANDARD_SHIPPING_COST_PER_UNIT
        )
        shipping_cost = dispatch_qty * unit_shipping_cost

        return dispatch_qty, shipping_cost, "dispatched"

    def _receive_arrivals(self, sim_day: int) -> None:
        """Move due shipments from transit into warehouse inventory."""

        remaining_shipments: List[TransitShipment] = []

        for shipment in self._in_transit_shipments:
            if shipment.eta_day <= sim_day:
                self._allocate_inbound_units(
                    product_id=shipment.product_id,
                    quantity=shipment.quantity,
                )
            else:
                remaining_shipments.append(shipment)

        self._in_transit_shipments = remaining_shipments

    def _allocate_inbound_units(self, product_id: str, quantity: int) -> None:
        """Distribute inbound units across warehouses via fixed deterministic ratios."""

        if quantity <= 0:
            return

        weights = ALLOCATION_WEIGHTS[product_id]
        distributed = 0

        for warehouse in WAREHOUSES[:-1]:
            units = int(quantity * weights[warehouse])
            self._inventory[warehouse][product_id] += units
            distributed += units

        # Assign any residual units to the final warehouse to preserve totals.
        residual = quantity - distributed
        self._inventory[WAREHOUSES[-1]][product_id] += residual

    def _fulfill_daily_demand(
        self,
        daily_demand: Dict[str, Dict[str, int]],
    ) -> Tuple[int, int]:
        """Serve local warehouse demand from local inventory only."""

        fulfilled_total = 0
        unmet_total = 0

        for warehouse in WAREHOUSES:
            for product in PRODUCTS:
                demand = int(daily_demand[warehouse][product])
                available = int(self._inventory[warehouse][product])

                fulfilled = min(available, demand)
                unmet = max(0, demand - fulfilled)

                self._inventory[warehouse][product] = available - fulfilled

                fulfilled_total += fulfilled
                unmet_total += unmet

        return fulfilled_total, unmet_total

    def _compute_daily_demand(self, day: int) -> Dict[str, Dict[str, int]]:
        """Compute deterministic daily demand with scenario crisis modifiers."""

        task = self._tasks[self._task_id]

        profile_index = (day - 1 + self._demand_profile_offset) % len(BASE_DEMAND_PROFILE)
        multiplier = BASE_DEMAND_PROFILE[profile_index]

        # Task 3 crisis trigger: demand spike starts on day 5.
        if self._task_id == "network-collapse" and day >= 5:
            multiplier *= 1.65

        demand: Dict[str, Dict[str, int]] = {}
        for warehouse in WAREHOUSES:
            demand[warehouse] = {}
            for product in PRODUCTS:
                base_value = task.base_daily_demand[warehouse][product]
                demand_value = int(round(base_value * multiplier))
                demand[warehouse][product] = max(0, demand_value)

        return demand

    def _effective_lead_time(
        self,
        supplier: SupplierConfig,
        transport_mode: TransportMode,
        sim_day: int,
    ) -> int:
        """Apply task-specific disruption effects to shipment lead time."""

        lead = supplier.air_lead_days if transport_mode == "AIR" else supplier.standard_lead_days

        if self._task_id == "minor-delay" and supplier.supplier_id == "sup_alpha":
            # Task 1 crisis: deterministic +2 day delay on the primary supplier.
            lead += 2

        if self._task_id == "network-collapse" and transport_mode == "STANDARD" and sim_day >= 5:
            # Severe network stress increases surface transport delays.
            lead += 1

        return max(1, lead)

    def _effective_supplier_capacity(self, supplier_id: str, sim_day: int) -> int:
        """Apply task-specific capacity shocks."""

        base_capacity = self._suppliers[supplier_id].daily_capacity

        if self._task_id == "network-collapse" and sim_day >= 5:
            # Task 3 crisis: raw material shortage cuts available supply.
            return max(0, int(base_capacity * 0.45))

        return base_capacity

    def _active_crisis_alerts_for_day(self, day: int) -> List[str]:
        """Return deterministic crisis alerts for the current task/day."""

        if self._task_id == "minor-delay":
            return [
                "minor-delay: sup_alpha shipments delayed by +2 days",
            ]

        if self._task_id == "port-strike":
            if 4 <= day <= 9:
                return [
                    "port-strike: major standard shipping lane blocked (days 4-9)",
                    "hint: use AIR selectively to avoid downstream stockouts",
                ]
            return [
                "port-strike: disruption window is days 4-9",
            ]

        # network-collapse
        if day >= 5:
            return [
                "network-collapse: demand spike active (+65%)",
                "network-collapse: raw material shortage active (capacity -55%)",
            ]
        return [
            "network-collapse: severe dual-shock starts at day 5",
        ]

    # -----------------------------
    # Deterministic scoring bounds
    # -----------------------------

    def _compute_task3_profit_bounds(self) -> Tuple[float, float]:
        """Compute deterministic baseline and optimistic profit bounds.

        This is intentionally a mathematical (closed-form) benchmark over the
        known 14-day demand schedule, as required by the task definition.
        """

        total_demand_units = 0

        # Use task3 demand schedule with crisis multipliers to compute aggregate demand.
        current_task = self._task_id
        self._task_id = "network-collapse"
        try:
            for day in range(1, MAX_EPISODE_DAYS + 1):
                daily = self._compute_daily_demand(day=day)
                for warehouse in WAREHOUSES:
                    for product in PRODUCTS:
                        total_demand_units += int(daily[warehouse][product])
        finally:
            self._task_id = current_task

        # Baseline policy assumptions (deterministic and conservative):
        # - 74% fulfillment under severe disruptions
        # - blended transport cost ~3.2 per demanded unit
        baseline_fill = 0.74
        baseline_fulfilled = total_demand_units * baseline_fill
        baseline_unmet = total_demand_units - baseline_fulfilled

        baseline_revenue = baseline_fulfilled * SELL_PRICE_PER_UNIT
        baseline_holding = total_demand_units * 0.18 * HOLDING_COST_PER_UNIT
        baseline_shipping = total_demand_units * 3.2
        baseline_penalty = baseline_unmet * STOCKOUT_PENALTY_PER_UNIT
        baseline_profit = baseline_revenue - baseline_holding - baseline_shipping - baseline_penalty

        # Optimistic upper bound assumptions:
        # - near-perfect fulfillment with efficient mixed transport
        oracle_fill = 0.97
        oracle_fulfilled = total_demand_units * oracle_fill
        oracle_unmet = total_demand_units - oracle_fulfilled

        oracle_revenue = oracle_fulfilled * SELL_PRICE_PER_UNIT
        oracle_holding = total_demand_units * 0.10 * HOLDING_COST_PER_UNIT
        oracle_shipping = total_demand_units * 1.9
        oracle_penalty = oracle_unmet * STOCKOUT_PENALTY_PER_UNIT
        oracle_profit = oracle_revenue - oracle_holding - oracle_shipping - oracle_penalty

        if oracle_profit <= baseline_profit:
            oracle_profit = baseline_profit + 1.0

        return float(baseline_profit), float(oracle_profit)

    # -----------------------------
    # Observation/state helpers
    # -----------------------------

    def _build_observation(self, current_day: int) -> ResilientOpsObservation:
        return ResilientOpsObservation(
            current_day=current_day,
            warehouse_inventory=self._copy_inventory(),
            in_transit_shipments=list(self._in_transit_shipments),
            active_crisis_alerts=self._active_crisis_alerts_for_day(max(1, current_day)),
            daily_demand=self._copy_demand(),
        )

    @staticmethod
    def _as_openenv_observation(result: ResilientOpsStepResult) -> ResilientOpsObservation:
        """Convert internal step result into OpenEnv observation contract."""

        return result.observation.model_copy(
            update={
                "done": result.done,
                "reward": float(result.reward.value),
                "metadata": dict(result.info),
            }
        )

    def _copy_inventory(self) -> Dict[str, Dict[str, int]]:
        return {
            warehouse: {product: int(value) for product, value in product_map.items()}
            for warehouse, product_map in self._inventory.items()
        }

    def _copy_demand(self) -> Dict[str, Dict[str, int]]:
        return {
            warehouse: {product: int(value) for product, value in product_map.items()}
            for warehouse, product_map in self._daily_demand.items()
        }

    def _total_inventory_units(self) -> int:
        total = 0
        for warehouse in WAREHOUSES:
            for product in PRODUCTS:
                total += int(self._inventory[warehouse][product])
        return total

    def _any_warehouse_out_of_stock(self) -> bool:
        """Return True if any warehouse has at least one product at zero units."""

        for warehouse in WAREHOUSES:
            for product in PRODUCTS:
                if int(self._inventory[warehouse][product]) <= 0:
                    return True
        return False

    @staticmethod
    def _normalize_daily_reward(raw_reward: float) -> float:
        normalized = raw_reward / DAILY_REWARD_NORMALIZATION_SCALE
        return max(-1.0, min(1.0, float(normalized)))

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(GRADER_EPSILON, min(1.0 - GRADER_EPSILON, float(value)))

    def close(self) -> None:
        """Clean up resources. Required by the OpenEnv server interface."""
        pass
__all__ = [
    "ResilientOpsAction",
    "ResilientOpsObservation",
    "ResilientOpsReward",
    "ResilientOpsStepResult",
    "ResilientOpsState",
    "ResilientOpsEnv",
]
