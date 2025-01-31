import torch


def normalize_to_neg_one_one(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Normalizes the input tensor `x` to the range [-1, 1] using scalar mean and std.

    Args:
    x (torch.Tensor): The tensor to normalize.
    mean (float): The mean computed from the dataset.
    std (float): The standard deviation computed from the dataset.

    Returns:
    torch.Tensor: The normalized tensor in the range [-1, 1].
    """
    std = std if std != 0 else 1  # Prevent division by zero
    x = (x - mean) / std  # Standardization
    return torch.tanh(x)  # Smoothly map to [-1, 1]


def denormalize_from_neg_one_one(
    x: torch.Tensor, mean: float, std: float
) -> torch.Tensor:
    """
    Denormalizes the input tensor `x` from the range [-1, 1] back to the original scale.

    Args:
    x (torch.Tensor): The tensor to denormalize (in the range [-1, 1]).
    mean (float): The mean used for normalization.
    std (float): The standard deviation used for normalization.

    Returns:
    torch.Tensor: The denormalized tensor.
    """
    x = torch.atanh(torch.clamp(x, -0.999, 0.999))  # Inverse tanh
    return x * std + mean  # Reverse standardization
