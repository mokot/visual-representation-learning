from itertools import product
from typing import List, Dict
from constants import GRID_SEARCH_SPACE


def generate_grid_search_combinations() -> List[Dict[str, int]]:
    keys = GRID_SEARCH_SPACE.keys()
    values = GRID_SEARCH_SPACE.values()
    combinations = list(product(*values))
    return [dict(zip(keys, combination)) for combination in combinations]
