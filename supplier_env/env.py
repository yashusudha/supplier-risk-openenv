import random
from typing import Dict, Tuple

from supplier_env.models import Observation, Action, Reward, SupplierInfo, StepInfo
from supplier_env.tasks import TaskConfig


class SupplierRiskEnv:
    def __init__(self, task: TaskConfig, seed: int = 42):
        self.task = task
        self.seed = seed
        self.rng = random.Random(seed)

        self.week = 0
        self.inventory = 0

        self.pending_orders: Dict[str, int] = {"A": 0, "B": 0, "C": 0}

        self.total_demand = 0
        self.total_fulfilled = 0
        self.total_cost = 0.0

        self.done = False

        # disruption flags (supplier shutdown)
        self.supplier_shutdown: Dict[str, bool] = {"A": False, "B": False, "C": False}

    def reset(self) -> Observation:
        self.week = 0
        self.inventory = self.task.initial_inventory

        self.pending_orders = {"A": 0, "B": 0, "C": 0}

        self.total_demand = 0
        self.total_fulfilled = 0
        self.total_cost = 0.0

        self.done = False

        self.supplier_shutdown = {"A": False, "B": False, "C": False}

        return self._get_observation()

    def state(self) -> Dict:
        return {
            "week": self.week,
            "inventory": self.inventory,
            "pending_orders": self.pending_orders,
            "total_demand": self.total_demand,
            "total_fulfilled": self.total_fulfilled,
            "total_cost": self.total_cost,
            "supplier_shutdown": self.supplier_shutdown,
        }

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.done:
            obs = self._get_observation()
            reward = Reward(
                value=0.0,
                revenue=0.0,
                ordering_cost=0.0,
                holding_cost=0.0,
                stockout_penalty=0.0,
                defect_penalty=0.0,
            )
            return obs, reward, True, {"msg": "Episode already finished"}

        self.week += 1

        # --- Disruption event: supplier shutdown ---
        if self.task.disruption_probability > 0:
            if self.rng.random() < self.task.disruption_probability:
                shutdown_supplier = self.rng.choice(["A", "B", "C"])
                self.supplier_shutdown[shutdown_supplier] = True

        # --- Receive deliveries from last week pending orders ---
        deliveries_received = {"A": 0, "B": 0, "C": 0}
        defects = {"A": 0, "B": 0, "C": 0}

        for supplier in ["A", "B", "C"]:
            qty = self.pending_orders[supplier]
            if qty <= 0:
                continue

            # Delay simulation
            delayed = self.rng.random() < self.task.supplier_delay_prob[supplier]
            if delayed:
                # delivery stays pending for next week again
                deliveries_received[supplier] = 0
            else:
                deliveries_received[supplier] = qty

                # Defects simulation
                defect_units = 0
                for _ in range(qty):
                    if self.rng.random() < self.task.supplier_defect_prob[supplier]:
                        defect_units += 1

                defects[supplier] = defect_units
                usable_units = qty - defect_units

                self.inventory += usable_units

            # clear pending if delivered, else keep it
            if not delayed:
                self.pending_orders[supplier] = 0

        # --- Apply warehouse capacity constraint ---
        if self.inventory > self.task.warehouse_capacity:
            self.inventory = self.task.warehouse_capacity

        # --- Generate demand ---
        demand = self.task.base_demand + self.rng.randint(
            -self.task.demand_variability, self.task.demand_variability
        )
        if demand < 0:
            demand = 0

        self.total_demand += demand

        # --- Fulfill demand ---
        fulfilled = min(self.inventory, demand)
        stockout = demand - fulfilled

        self.inventory -= fulfilled
        self.total_fulfilled += fulfilled

        # --- Revenue ---
        revenue = fulfilled * self.task.revenue_per_unit

        # --- Ordering costs (for new orders placed this step) ---
        ordering_cost = 0.0
        new_orders = {"A": action.order_A, "B": action.order_B, "C": action.order_C}

        for supplier, qty in new_orders.items():
            if qty < 0:
                qty = 0

            # supplier shutdown means cannot order
            if self.supplier_shutdown[supplier]:
                qty = 0

            # capacity constraint
            qty = min(qty, self.task.supplier_max_capacity[supplier])

            ordering_cost += qty * self.task.supplier_costs[supplier]

            # store into pending (arrives next week if not delayed)
            self.pending_orders[supplier] += qty

        self.total_cost += ordering_cost

        # --- Holding cost ---
        holding_cost = self.inventory * self.task.holding_cost_per_unit

        # --- Stockout penalty ---
        stockout_penalty = stockout * self.task.stockout_penalty_per_unit

        # --- Defect penalty ---
        defect_penalty = sum(defects.values()) * self.task.defect_penalty_per_unit

        # --- Total reward ---
        reward_value = revenue - ordering_cost - holding_cost - stockout_penalty - defect_penalty

        reward = Reward(
            value=reward_value,
            revenue=revenue,
            ordering_cost=ordering_cost,
            holding_cost=holding_cost,
            stockout_penalty=stockout_penalty,
            defect_penalty=defect_penalty,
        )

        # Episode done check
        if self.week >= self.task.max_weeks:
            self.done = True

        obs = self._get_observation()

        info_obj = StepInfo(
            demand=demand,
            fulfilled=fulfilled,
            stockout=stockout,
            defects=defects,
            deliveries_received=deliveries_received,
        )

        return obs, reward, self.done, info_obj.model_dump()

    def _get_observation(self) -> Observation:
        suppliers_data = {}
        for s in ["A", "B", "C"]:
            suppliers_data[s] = SupplierInfo(
                cost_per_unit=self.task.supplier_costs[s],
                delay_probability=self.task.supplier_delay_prob[s],
                defect_probability=self.task.supplier_defect_prob[s],
                max_capacity=self.task.supplier_max_capacity[s],
            )

        forecast = self.task.base_demand

        return Observation(
            week=self.week,
            inventory=self.inventory,
            demand_forecast=forecast,
            suppliers=suppliers_data,
            pending_orders=self.pending_orders.copy(),
            total_demand_so_far=self.total_demand,
            total_fulfilled_so_far=self.total_fulfilled,
            total_cost_so_far=self.total_cost,
        )