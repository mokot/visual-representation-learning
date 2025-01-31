import torch


def normalize_to_neg_one_one(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """
    Normalizes the input tensor `x` to the range [-1, 1] using the provided mean and std.

    Args:
    x (torch.Tensor): The tensor to normalize.
    mean (torch.Tensor): The mean computed from the dataset.
    std (torch.Tensor): The standard deviation computed from the dataset.

    Returns:
    torch.Tensor: The normalized tensor in the range [-1, 1].
    """
    std[std == 0] = 1  # Avoid division by zero
    x = (x - mean) / std  # Standardization
    return torch.tanh(x)  # Map to [-1, 1]


def denormalize_from_neg_one_one(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """
    Denormalizes the input tensor `x` from the range [-1, 1] back to the original distribution.

    Args:
    x (torch.Tensor): The tensor to denormalize (in the range [-1, 1]).
    mean (torch.Tensor): The mean used for normalization.
    std (torch.Tensor): The standard deviation used for normalization.

    Returns:
    torch.Tensor: The denormalized tensor in the original range.
    """
    x = torch.atanh(torch.clamp(x, -0.999, 0.999))  # Inverse tanh
    return x * std + mean  # Reverse standardization
