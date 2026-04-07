from pydantic import BaseModel, Field
from typing import Dict, List


class SupplierInfo(BaseModel):
    cost_per_unit: float = Field(..., ge=0)
    delay_probability: float = Field(..., ge=0.0, le=1.0)
    defect_probability: float = Field(..., ge=0.0, le=1.0)
    max_capacity: int = Field(..., ge=0)


class Observation(BaseModel):
    week: int = Field(..., ge=0)
    inventory: int = Field(..., ge=0)

    demand_forecast: int = Field(..., ge=0)

    suppliers: Dict[str, SupplierInfo]

    pending_orders: Dict[str, int]  # supplier -> arriving quantity next week

    total_demand_so_far: int = Field(..., ge=0)
    total_fulfilled_so_far: int = Field(..., ge=0)
    total_cost_so_far: float = Field(..., ge=0.0)


class Action(BaseModel):
    order_A: int = Field(..., ge=0)
    order_B: int = Field(..., ge=0)
    order_C: int = Field(..., ge=0)


class Reward(BaseModel):
    value: float
    revenue: float
    ordering_cost: float
    holding_cost: float
    stockout_penalty: float
    defect_penalty: float


class StepInfo(BaseModel):
    demand: int
    fulfilled: int
    stockout: int
    defects: Dict[str, int]
    deliveries_received: Dict[str, int]