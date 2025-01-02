import torch
from PIL import Image
from pathlib import Path
from constants import CIFAR10_TRANSFORM


def image_path_to_tensor(image_path: Path) -> torch.Tensor:
    """
    Converts an image file to a tensor using the CIFAR10_TRANSFORM pipeline.

    Args:
    - image_path (Path): The path to the image file.

    Returns:
    - torch.Tensor: The processed image as a tensor.
    """
    try:
        # Open the image and apply the CIFAR10 transform
        image = Image.open(image_path).convert("RGB")  # Ensure it's in RGB format
        return CIFAR10_TRANSFORM(image)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")


def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    """
    Preprocesses the image tensor for model input.

    Args:
    - image (torch.Tensor): The image tensor to preprocess.

    Returns:
    - torch.Tensor: The preprocessed image tensor on the appropriate device.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a torch.Tensor.")

    # Normalize if the tensor is in the range [0, 255]
    if image.max() > 1.0:
        image = image / 255.0

    # Move the tensor to the appropriate device
    return image.to(
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
