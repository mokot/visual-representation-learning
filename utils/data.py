import io
import torch
from PIL import Image
from pathlib import Path
from torchvision import datasets
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from constants import CIFAR10_TRANSFORM
from utils.image import tensor_to_image

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


def save_gs_data(
    image: Image.Image,
    label: int,
    splat: torch.nn.ParameterDict,
    file_path: Path = Path("data.pt"),
) -> None:
    """
    Saves the image, label, and splat to a file.

    Args:
        image (Image.Image): A PIL image.
        label (int): An integer label.
        splat (torch.nn.ParameterDict): A ParameterDict object.
        file_path (Path): Path to save the data. Default is 'data.pt'.
    """
    # Ensure the parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert PIL Image to bytes for serialization
    img_buffer = io.BytesIO()
    image = tensor_to_image(image)
    image.save(img_buffer, format="PNG")  # Change format as needed
    img_buffer.seek(0)

    torch.save(
        {
            "image": img_buffer.getvalue(),  # Save image as bytes
            "label": label,
            "splat": splat.state_dict(),  # Save state_dict for ParameterDict
        },
        str(file_path),  # Convert Path to string for torch.save
    )


def load_gs_data(
    file_path: Path = Path("data.pt"),
) -> Tuple[Image.Image, int, torch.nn.ParameterDict]:
    """
    Loads the image, label, and splat from a file.

    Args:
        file_path (Path): Path to load the data from. Default is 'data.pt'.

    Returns:
        Tuple[Image.Image, int, torch.nn.ParameterDict]: The loaded image, label, and splat ParameterDict.
    """
    data = torch.load(
        str(file_path), weights_only=False
    )  # Convert Path to string for torch.load

    # Load image from bytes
    img_buffer = io.BytesIO(data["image"])
    image = Image.open(img_buffer)

    # Create ParameterDict with the same structure as the saved one
    splat = torch.nn.ParameterDict(
        {
            key: torch.nn.Parameter(torch.empty_like(value))
            for key, value in data["splat"].items()
        }
    )
    splat.load_state_dict(data["splat"])

    return image, data["label"], splat
