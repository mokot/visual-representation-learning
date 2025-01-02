import torch
import numpy as np
from PIL import Image
from typing import List
from pathlib import Path
from constants.transforms import CIFAR10_INVERSE_TRANSFORM


def save_gif(frames: List[torch.Tensor], results_path: Path) -> None:
    """
    Saves a list of frames as a GIF.

    Args:
    - frames (List[torch.Tensor]): List of image tensors with shape (H, W, C) or (C, H, W).
    - results_path (Path): The relative path to save the GIF.

    Returns:
    - None
    """
    # Convert tensors to PIL images
    pil_frames = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            if frame.ndim == 3 and frame.shape[0] == 3:  # (C, H, W)
                frame = frame.permute(1, 2, 0)  # Convert to (H, W, C)
            frame = (frame.cpu().detach().numpy() * 255).astype("uint8")
            pil_frames.append(Image.fromarray(frame))
        elif isinstance(frame, np.ndarray):
            pil_frames.append(Image.fromarray(frame))
        elif isinstance(frame, Image.Image):
            pil_frames.append(frame)
        else:
            raise ValueError(
                "Frames must be either torch.Tensor or PIL.Image.Image objects."
            )

    # Save as GIF
    gif_path = results_path.with_suffix(".gif")
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=False,
        duration=5,
        loop=0,
    )
    print(f"GIF saved to: {gif_path}")


def save_tensor(tensor: torch.Tensor, results_path: Path) -> None:
    """
    Saves a tensor as a JPG image.

    Args:
    - tensor (torch.Tensor): The image tensor with shape (H, W, C) or (C, H, W).
    - results_path (Path): The relative path to save the image.

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
        raise RuntimeError(f"Error transforming from tensor: {e}")

    # Save as JPG
    jpg_path = results_path.with_suffix(".jpg")
    pil_image.save(jpg_path, format="JPEG")
    print(f"JPG image saved to: {jpg_path}")
