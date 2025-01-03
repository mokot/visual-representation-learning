import torch
import random
import numpy as np


def set_random_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility.

    Args:
    - seed (int): The random seed.

    Returns:
    - None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
