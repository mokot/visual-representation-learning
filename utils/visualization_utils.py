import torch
from pathlib import Path
from IPython.display import Image as IPImage, display
from constants import CIFAR10_INVERSE_TRANSFORM


def visualize_gif(gif_path: Path) -> None:
    """
    Visualizes the GIF at the given path.

    Args:
    - gif_path (Path): The path to the GIF.

    Returns:
    - None
    """
    if not gif_path.exists() or not gif_path.is_file():
        raise FileNotFoundError(f"GIF not found at: {gif_path}")

    try:
        display(IPImage(filename=gif_path))
    except Exception as e:
        raise RuntimeError(f"Error displaying GIF: {e}")


def visualize_tensor(tensor: torch.Tensor) -> None:
    """
    Reverses the tensor transformation and visualizes the image as a PIL image.

    Args:
    - tensor (torch.Tensor): The input tensor with shape (C, H, W) normalized between [-1, 1].

    Returns:
    - None
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor must be a torch.Tensor.")

    if len(tensor.shape) != 3 or tensor.shape[0] != 3:
        raise ValueError("Tensor must have shape (C, H, W) with 3 color channels.")

    try:
        pil_image = CIFAR10_INVERSE_TRANSFORM(tensor)
        pil_image.show()
    except Exception as e:
        raise RuntimeError(f"Error visualizing tensor: {e}")
