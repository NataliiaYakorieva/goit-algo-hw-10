import pulp
from typing import Dict


def optimize_production(resource_limits: Dict[str, int]) -> Dict[str, int]:
    """
    Solves the production optimization problem for Lemonade and Fruit Juice using PuLP.

    Args:
        resource_limits (dict): Dictionary with resource names and their available quantities.
            Keys: 'water', 'sugar', 'lemon_juice', 'fruit_puree'

    Returns:
        dict: Optimal quantities of Lemonade, Fruit Juice, and total products.
    """
    model: pulp.LpProblem = pulp.LpProblem(
        "Maximize_Production", pulp.LpMaximize)

    lemonade: pulp.LpVariable = pulp.LpVariable(
        'Lemonade', lowBound=0, cat='Integer')
    fruit_juice: pulp.LpVariable = pulp.LpVariable(
        'FruitJuice', lowBound=0, cat='Integer')

    model += lemonade + fruit_juice, "Total_Products"

    # Resource constraints
    model += 2 * lemonade + 1 * \
        fruit_juice <= resource_limits['water'], "Water"
    model += 1 * lemonade <= resource_limits['sugar'], "Sugar"
    model += 1 * lemonade <= resource_limits['lemon_juice'], "LemonJuice"
    model += 2 * fruit_juice <= resource_limits['fruit_puree'], "FruitPuree"

    # Solve the problem
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    result = {
        "lemonade": int(lemonade.value()),
        "fruit_juice": int(fruit_juice.value()),
        "total_products": int(lemonade.value() + fruit_juice.value())
    }
    return result


if __name__ == "__main__":
    # Example 1: Original resource limits
    limits1 = {
        "water": 100,
        "sugar": 50,
        "lemon_juice": 30,
        "fruit_puree": 40
    }
    result1 = optimize_production(limits1)
    print("Example 1 result:", result1)

    # Example 2: Different resource limits
    limits2 = {
        "water": 80,
        "sugar": 20,
        "lemon_juice": 10,
        "fruit_puree": 50
    }
    result2 = optimize_production(limits2)
    print("Example 2 result:", result2)
