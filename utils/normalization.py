import torch


def normalize_to_neg_one_one(
    x: torch.Tensor, min_val: float, max_val: float
) -> torch.Tensor:
    """
    Normalizes the input tensor `x` to the range [-1, 1] based on the given `min_val` and `max_val`.

    Args:
    x (torch.Tensor): The tensor to normalize.
    min_val (float): The minimum value of the original range.
    max_val (float): The maximum value of the original range.

    Returns:
    torch.Tensor: The normalized tensor in the range [-1, 1].
    """
    x = 2 * (x - min_val) / (max_val - min_val) - 1
    x[x != x] = 0
    return x


def denormalize_from_neg_one_one(
    x: torch.Tensor, min_val: float, max_val: float
) -> torch.Tensor:
    """
    Denormalizes the input tensor `x` from the range [-1, 1] back to the original range defined by `min_val` and `max_val`.

    Args:
    x (torch.Tensor): The tensor to denormalize (in the range [-1, 1]).
    min_val (float): The minimum value of the original range.
    max_val (float): The maximum value of the original range.

    Returns:
    torch.Tensor: The denormalized tensor in the original range [min_val, max_val].
    """
    x = (x + 1) / 2 * (max_val - min_val) + min_val
    x[x != x] = 0
    return x
