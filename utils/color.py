import torch


def convert_rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """
    Converts an RGB tensor to spherical harmonics (SH) space.

    Args:
        rgb (torch.Tensor): Input tensor of shape (..., 3) representing RGB values.

    Returns:
        torch.Tensor: Transformed tensor in SH space.
    """
    C0 = 0.28209479177387814  # Normalization constant for SH basis
    return (rgb - 0.5) / C0
