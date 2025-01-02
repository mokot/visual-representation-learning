import torch
from pathlib import Path
from constants import CIFAR10_INVERSE_TRANSFORM
from IPython.display import Image as IPImage, display, HTML


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

    if len(tensor.shape) != 3 or tensor.shape[2] != 3:
        raise ValueError("Tensor must have shape (C, H, W) with 3 color channels.")

    try:
        pil_image = CIFAR10_INVERSE_TRANSFORM(tensor)
        pil_image.show()
    except Exception as e:
        raise RuntimeError(f"Error visualizing tensor: {e}")


def visualize_results(
    animation_path: Path, final_image_path: Path, original_image_path: Path
) -> None:
    """
    Visualizes an animation (GIF), a final image (JPG), and an original image (JPG).

    Args:
    - animation_path (Path): Path to the animation (GIF) file.
    - final_image_path (Path): Path to the final JPG image file.
    - original_image_path (Path): Path to the original JPG image file.

    Returns:
    - None
    """
    # Create HTML for displaying images in a row and display it
    html_content = f"""
    <div style="display: flex; align-items: center; justify-content: center;">
        <div style="margin: 0px 25px; text-align: center;">
            <h3>Animation</h3>
            <img src="{animation_path}" style="min-width: 100px; max-width: 100px;">
        </div>
        <div style="margin: 0px 25px; text-align: center;">
            <h3>Final Image</h3>
            <img src="{final_image_path}" style="min-width: 100px; max-width: 100px;">
        </div>
        <div style="margin: 0px 25px; text-align: center;">
            <h3>Original Image</h3>
            <img src="{original_image_path}" style="min-width: 100px; max-width: 100px;">
        </div>
    </div>
    """
    display(HTML(html_content))
