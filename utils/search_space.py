from itertools import product
from typing import Dict, List, Any, Type


def is_valid_combination(combination: Dict[str, Any], trainer_class: Type) -> bool:
    """
    Validates a combination of parameters for a given trainer class.

    Args:
        combination (Dict[str, Any]): The parameter combination to validate.
        trainer_class (Type): The trainer class being validated.

    Returns:
        bool: True if the combination is valid, False otherwise.
    """
    if trainer_class.__name__ != "GaussianImageTrainer":
        return True

    # Default values are set from config.py
    try:
        if combination.get("learning_rate", None) and not combination.get(
            "group_optimization", True
        ):
            return False
        if sum(combination.get("loss_weights", [1 / 3, 1 / 3, 1 / 3])) != 1:
            return False
        if (
            combination.get("distortion_loss_weight", None)
            or combination.get("normal_loss_weight", None)
        ) and combination.get("model_type", "2dgs") == "3dgs":
            return False
        if combination.get("strategy", None) and combination.get(
            "group_optimization", True
        ):
            return False
        if (
            combination.get("selective_adam", False)
            and combination.get("model_type", False) != "3dgs"
        ):
            return False
        if (
            combination.get("bilateral_grid", False)
            and combination.get("model_type", "2dgs") != "3dgs"
        ):
            return False
    except KeyError as e:
        print(f"Missing required key: {e}")
        return False
    except TypeError as e:
        print(f"Invalid parameter type: {e}")
        return False

    return True


def generate_grid_search_combinations(
    grid_search_space: Dict[str, List[Any]], trainer_class: Type
) -> List[Dict[str, Any]]:
    """
    Generate all valid combinations for grid search based on the given search space and a validation function.

    Args:
        grid_search_space (Dict[str, List[Any]]): A dictionary where keys are parameter names
            and values are lists of possible values for each parameter.
        trainer_class (Type): A reference to the trainer class used for validation.

    Returns:
        List[Dict[str, Any]]: A list of valid parameter combinations as dictionaries.
    """
    keys = list(grid_search_space.keys())
    values = list(grid_search_space.values())

    # Generate all combinations using itertools.product
    combinations = product(*values)

    # Create a list of valid combinations
    all_combinations = [
        dict(zip(keys, combination))
        for combination in combinations
        if is_valid_combination(dict(zip(keys, combination)), trainer_class)
    ]

    return all_combinations
