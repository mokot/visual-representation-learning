import torch
from pathlib import Path
from typing import Optional, Literal
from gaussian_image_trainer import GaussianImageTrainer
from image_utils import image_path_to_tensor, preprocess_image
from data_utils import create_default_image


def train(
    height: int = 32,
    width: int = 32,
    num_points: int = 1024,
    save_imgs: bool = True,
    img: Optional[torch.Tensor] = None,
    img_path: Optional[Path] = None,
    iterations: int = 1000,
    lr: float = 0.01,
    model_type: Literal["3dgs", "2dgs"] = "3dgs",
) -> None:
    """
    Trains random Gaussians to fit an image.

    Args:
    - height (int): The height of the image.
    - width (int): The width of the image.
    - num_points (int): The number of Gaussians to use.
    - save_imgs (bool): Whether to save the images during training.
    - img (Optional[torch.Tensor]): The image tensor to fit.
    - img_path (Optional[Path]): The path to the image to fit.
    - iterations (int): The number of iterations to train.
    - lr (float): The learning rate for optimization.
    - model_type (Literal["3dgs", "2dgs"]): The model type to use.

    Raises:
    - ValueError: If both `img` and `img_path` are provided.
    - Exception: If training fails due to other errors.
    """
    # Validate inputs
    if img is not None and img_path is not None:
        raise ValueError("Provide either 'img' or 'img_path', not both.")

    # Load or create the ground truth image
    if img is not None:
        gt_image = img
    elif img_path is not None:
        if not img_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {img_path}")
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = create_default_image(height=height, width=width)

    # Preprocess the image
    gt_image = preprocess_image(gt_image)

    # Initialize the Gaussian Image Trainer
    trainer = GaussianImageTrainer(gt_image=gt_image, num_points=num_points)

    # Train the model
    try:
        trainer.train(
            iterations=iterations,
            lr=lr,
            save_imgs=save_imgs,
            model_type=model_type,
        )
    except Exception as e:
        print(f"Training failed due to: {e}")
        raise
