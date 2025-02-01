import torch


def normalize_to_neg_one_one(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    min_val: float,
    max_val: float,
) -> torch.Tensor:
    """
    Normalizes the input tensor `x` to the range [-1, 1] using the provided mean and std.

    Args:
    x (torch.Tensor): The tensor to normalize.
    mean (torch.Tensor): The mean computed from the dataset.
    std (torch.Tensor): The standard deviation computed from the dataset.
    min_val (float): The minimum value of the range.
    max_val (float): The maximum value of the range.

    Returns:
    torch.Tensor: The normalized tensor in the range [-1, 1].
    """
    # Move to the correct device
    mean = mean.to(x.device)
    std = std.to(x.device)

    std[std == 0] = 1  # Avoid division by zero
    x = (x - mean) / std  # Standardization

    # Map to [-1, 1]
    return 2 * (x - min_val) / (max_val - min_val) - 1


def denormalize_from_neg_one_one(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    min_val: float,
    max_val: float,
) -> torch.Tensor:
    """
    Denormalizes the input tensor `x` from the range [-1, 1] back to the original distribution.

    Args:
    x (torch.Tensor): The tensor to denormalize (in the range [-1, 1]).
    mean (torch.Tensor): The mean used for normalization.
    std (torch.Tensor): The standard deviation used for normalization.
    min_val (float): The minimum value of the range.
    max_val (float): The maximum value of the range.

    Returns:
    torch.Tensor: The denormalized tensor in the original range.
    """
    # Move to the correct device
    mean = mean.to(x.device)
    std = std.to(x.device)

    # Map from [-1, 1] to [min_val, max_val]
    x = (x + 1) / 2 * (max_val - min_val) + min_val
    return x * std + mean  # Reverse standardization
