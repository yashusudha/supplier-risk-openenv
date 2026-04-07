from dataclasses import dataclass
from typing import Dict


@dataclass
class TaskConfig:
    name: str
    max_weeks: int

    initial_inventory: int
    warehouse_capacity: int

    base_demand: int
    demand_variability: int  # random fluctuation range

    revenue_per_unit: float
    holding_cost_per_unit: float
    stockout_penalty_per_unit: float
    defect_penalty_per_unit: float

    supplier_costs: Dict[str, float]
    supplier_delay_prob: Dict[str, float]
    supplier_defect_prob: Dict[str, float]
    supplier_max_capacity: Dict[str, int]

    disruption_probability: float  # chance supplier shutdown event happens


def get_easy_task() -> TaskConfig:
    return TaskConfig(
        name="easy",
        max_weeks=20,
        initial_inventory=200,
        warehouse_capacity=1000,
        base_demand=120,
        demand_variability=20,
        revenue_per_unit=15.0,
        holding_cost_per_unit=0.5,
        stockout_penalty_per_unit=20.0,
        defect_penalty_per_unit=10.0,
        supplier_costs={"A": 6.0, "B": 8.0, "C": 5.0},
        supplier_delay_prob={"A": 0.10, "B": 0.05, "C": 0.15},
        supplier_defect_prob={"A": 0.05, "B": 0.02, "C": 0.08},
        supplier_max_capacity={"A": 200, "B": 150, "C": 250},
        disruption_probability=0.0,
    )


def get_medium_task() -> TaskConfig:
    return TaskConfig(
        name="medium",
        max_weeks=25,
        initial_inventory=150,
        warehouse_capacity=900,
        base_demand=150,
        demand_variability=50,
        revenue_per_unit=15.0,
        holding_cost_per_unit=0.7,
        stockout_penalty_per_unit=25.0,
        defect_penalty_per_unit=12.0,
        supplier_costs={"A": 6.5, "B": 9.0, "C": 5.5},
        supplier_delay_prob={"A": 0.20, "B": 0.08, "C": 0.25},
        supplier_defect_prob={"A": 0.08, "B": 0.03, "C": 0.12},
        supplier_max_capacity={"A": 220, "B": 160, "C": 260},
        disruption_probability=0.05,
    )


def get_hard_task() -> TaskConfig:
    return TaskConfig(
        name="hard",
        max_weeks=30,
        initial_inventory=100,
        warehouse_capacity=800,
        base_demand=170,
        demand_variability=80,
        revenue_per_unit=15.0,
        holding_cost_per_unit=1.0,
        stockout_penalty_per_unit=35.0,
        defect_penalty_per_unit=15.0,
        supplier_costs={"A": 7.0, "B": 10.0, "C": 6.0},
        supplier_delay_prob={"A": 0.30, "B": 0.12, "C": 0.35},
        supplier_defect_prob={"A": 0.12, "B": 0.05, "C": 0.18},
        supplier_max_capacity={"A": 200, "B": 140, "C": 230},
        disruption_probability=0.15,
    )


def get_task(task_name: str) -> TaskConfig:
    if task_name == "easy":
        return get_easy_task()
    elif task_name == "medium":
        return get_medium_task()
    elif task_name == "hard":
        return get_hard_task()
    else:
        raise ValueError(f"Unknown task: {task_name}")