"""
Pareto frontier and dominance calculation engine for efficiency-reliability trade-offs.
"""

from typing import Any, Dict, List, Tuple


def is_pareto_dominant(
    candidate: Dict[str, Any],
    other: Dict[str, Any],
    objectives: List[Tuple[str, str]],
) -> bool:
    """
    Evaluates whether candidate strictly dominates other across mixed objectives.

    candidate dominates other iff:
    - candidate is at least as good as other on every objective:
        val_cand <= val_other for minimization
        val_cand >= val_other for maximization
    - candidate is strictly better than other on at least one objective:
        val_cand < val_other for minimization
        val_cand > val_other for maximization

    Args:
        candidate: Dictionary containing metric keys.
        other: Dictionary containing metric keys.
        objectives: List of tuples (metric_key, direction), where direction is 'min' or 'max'.

    Returns:
        True if candidate dominates other, False otherwise.
    """
    at_least_as_good = True
    strictly_better = False

    for key, direction in objectives:
        val_cand = candidate[key]
        val_other = other[key]

        if val_cand is None or val_other is None:
            return False

        if direction == "min":
            if val_cand > val_other:
                at_least_as_good = False
                break
            if val_cand < val_other:
                strictly_better = True
        elif direction == "max":
            if val_cand < val_other:
                at_least_as_good = False
                break
            if val_cand > val_other:
                strictly_better = True
        else:
            raise ValueError(f"Unknown direction '{direction}', must be 'min' or 'max'")

    return at_least_as_good and strictly_better


def compute_pareto_frontier(
    items: List[Dict[str, Any]],
    objectives: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Identifies Pareto-efficient (non-dominated) and dominated items.

    Args:
        items: List of dictionaries representing configurations.
        objectives: List of (metric_key, direction), where direction is 'min' or 'max'.

    Returns:
        Dict with keys:
            - 'frontier': List of non-dominated items.
            - 'dominated': List of dominated items with their dominating items.
            - 'all_items': List of items tagged with 'is_pareto_efficient': bool.
    """
    frontier = []
    dominated = []
    annotated_items = []

    for item in items:
        # Check if item is dominated by any other item
        is_dom = False
        dominating_items = []
        for other in items:
            if item is other:
                continue
            if is_pareto_dominant(other, item, objectives):
                is_dom = True
                dominating_items.append(other)

        item_copy = dict(item)
        item_copy["is_pareto_efficient"] = not is_dom
        if is_dom:
            item_copy["dominated_by"] = [
                f"{d.get('model', '')}_{d.get('precision', '')}" for d in dominating_items
            ]
            dominated.append(item_copy)
        else:
            frontier.append(item_copy)
        annotated_items.append(item_copy)

    # Sort frontier by first objective
    first_key, first_dir = objectives[0]
    reverse = (first_dir == "max")
    frontier.sort(key=lambda x: x[first_key] if x[first_key] is not None else 0, reverse=reverse)

    return {
        "frontier": frontier,
        "dominated": dominated,
        "all_items": annotated_items,
    }

