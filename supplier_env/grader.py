from supplier_env.env import SupplierRiskEnv


def clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def grade_episode(env: SupplierRiskEnv) -> float:
    """
    Returns score between 0.0 and 1.0
    Based on service level + cost efficiency.
    """

    state = env.state()

    total_demand = state["total_demand"]
    total_fulfilled = state["total_fulfilled"]
    total_cost = state["total_cost"]

    if total_demand == 0:
        service_level = 1.0
    else:
        service_level = total_fulfilled / total_demand

    # Cost efficiency normalization (rough scaling)
    # Lower cost is better. We convert to score using exponential-ish scaling.
    # This is deterministic and stable.
    cost_score = 1.0 / (1.0 + (total_cost / 5000.0))

    score = 0.7 * service_level + 0.3 * cost_score

    return clamp(score)