import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from constants import CIFAR10_TRANSFORM


def load_cifar10(
    batch_size: int = 64,
    shuffle: bool = True,
    train: bool = True,
    data_root: str = "./data",
) -> DataLoader:
    """
    Loads the CIFAR-10 dataset.

    Args:
    - batch_size (int): The number of samples per batch.
    - shuffle (bool): Whether to shuffle the dataset.
    - train (bool): Whether to load the training or test set.
    - data_root (str): Root directory for downloading/storing the dataset.

    Returns:
    - DataLoader: An iterable over the dataset.
    """
    dataset = datasets.CIFAR10(
        root=data_root,
        train=train,
        download=True,
        transform=CIFAR10_TRANSFORM,  # Use the transform defined in constants
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,  # Use 2 worker threads for faster data loading
    )

    return dataloader


def create_default_image(height: int = 32, width: int = 32) -> torch.Tensor:
    """
    Creates a default image with red in the top left and blue in the bottom right.

    Args:
    - height (int): The height of the image.
    - width (int): The width of the image.

    Returns:
    - torch.Tensor: The default image tensor with shape (height, width, 3).
    """
    image = torch.ones(
        (height, width, 3), dtype=torch.float32
    )  # Ensure float32 for compatibility
    image[: height // 2, : width // 2, :] = torch.tensor(
        [1.0, 0.0, 0.0], dtype=torch.float32
    )
    image[height // 2 :, width // 2 :, :] = torch.tensor(
        [0.0, 0.0, 1.0], dtype=torch.float32
    )
    return image
