import torch
from pathlib import Path
from typing import Optional, Literal
from models.gaussian_image_trainer_bak import GaussianImageTrainer
from utils import image_path_to_tensor, create_default_image, preprocess_image


def train(
    height: int = 32,
    width: int = 32,
    num_points: int = 1024,
    results_path: Optional[Path] = None,
    image: Optional[torch.Tensor] = None,
    image_path: Optional[Path] = None,
    iterations: int = 1000,
    lr: float = 0.01,
    model_type: Literal["2dgs", "3dgs"] = "2dgs",
) -> None:
    """
    Trains random Gaussians to fit an image.

    Args:
    - height (int): The height of the image.
    - width (int): The width of the image.
    - num_points (int): The number of Gaussians to use.
    - results_path (Optional[Path]): The path to save the results.
    - image (Optional[torch.Tensor]): The image tensor to fit.
    - image_path (Optional[Path]): The path to the image to fit.
    - iterations (int): The number of iterations to train.
    - lr (float): The learning rate for optimization.
    - model_type (Literal["2dgs", "3dgs"]): The model type to use.

    Raises:
    - ValueError: If both `image` and `image_path` are provided.
    - Exception: If training fails due to other errors.
    """
    # Check if training is possible
    if not torch.cuda.is_available():
        print("No GPU available. `gpsplat` requires a GPU to train.")
        return
    device = torch.device("cuda")

    # Validate inputs
    if image is not None and image_path is not None:
        raise ValueError("Provide either 'image' or 'image_path', not both.")

    # Load or create the ground truth image
    if image is not None:
        gt_image = image
    elif image_path is not None:
        if not image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        gt_image = image_path_to_tensor(image_path)
    else:
        gt_image = create_default_image(height=height, width=width)

    # Preprocess the image
    gt_image = preprocess_image(gt_image, device)

    # Initialize the Gaussian Image Trainer
    trainer = GaussianImageTrainer(gt_image=gt_image, num_points=num_points)

    # Train the model
    try:
        trainer.train(
            iterations=iterations,
            lr=lr,
            results_path=results_path,
            model_type=model_type,
        )
    except Exception as e:
        print(f"Training failed due to: {e}")
        raise
