import torch
from pathlib import Path
from typing import Dict, List
from torchvision import datasets
from torch.utils.data import DataLoader
from constants import CIFAR10_TRANSFORM


def load_cifar10(
    batch_size: int = 64,
    shuffle: bool = True,
    train: bool = True,
    data_root: Path = Path("data"),
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
        root=f"{data_root}/{'train' if train else 'test'}",
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
    image = (image - 0.5) / 0.5  # Normalize to mean 0 and variance 1
    return image


def collect_class_images(
    dataloader: torch.utils.data.DataLoader, N: int
) -> Dict[str, List[torch.Tensor]]:
    """
    Collects N images for each class from the provided DataLoader.

    Args:
    - dataloader (torch.utils.data.DataLoader): The DataLoader containing the dataset.
    - N (int): The number of images to collect for each class.

    Returns:
    - Dict[str, List[torch.Tensor]]: A dictionary where keys are class names and values are lists of image tensors.
    """
    class_images = {k: [] for k in dataloader.dataset.class_to_idx.keys()}

    # Iterate over the DataLoader and collect images
    for images, labels in dataloader:
        for image, label in zip(images, labels):
            label_name = dataloader.dataset.classes[label.item()]

            if len(class_images[label_name]) < N:
                class_images[label_name].append(image)

            if all(len(v) == N for v in class_images.values()):
                return class_images

    return class_images
