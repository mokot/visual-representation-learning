from itertools import product
from typing import List, Dict


def generate_grid_search_combinations(
    grid_search_space: Dict[str, object]
) -> List[Dict[str, int]]:
    keys = grid_search_space.keys()
    values = grid_search_space.values()
    combinations = list(product(*values))
    return [dict(zip(keys, combination)) for combination in combinations]
