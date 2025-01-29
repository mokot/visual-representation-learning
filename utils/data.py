import io
import torch
from PIL import Image
from pathlib import Path
from torchvision import datasets
from torch.utils.data import DataLoader
from constants import CIFAR10_TRANSFORM
from utils.image import tensor_to_image
from typing import Any, Dict, List, Tuple
from constants.splats import CIFAR10_KS, CIFAR10_VIEWMATS
from utils.normalization import normalize_to_neg_one_one, denormalize_from_neg_one_one


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
        str(file_path), weights_only=False, map_location=torch.device("cpu")
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

    # Rename color to colors
    if "color" in splat:
        splat["colors"] = splat.pop("color")

    return image, data["label"], splat


def transform_autoencoder_input(
    parameter_dict: Dict[str, torch.Tensor], join_mode: str = "flatten"
) -> torch.Tensor | Dict[str, torch.Tensor]:
    """
    Transforms the Gaussian splat parameters into a flattened input format suitable for an autoencoder.

    Args:
        parameter_dict (dict): Dictionary containing Gaussian splat parameters.
        join_mode (str): Determines how to concatenate tensors:
            - "flatten": Flatten all tensors and concatenate them, result is a 1D tensor.
            - "concat": Concatenate all tensors along the channel, result is a 2D tensor (32x32xN).
            - "dict": Converts to 2D tensor and returns as a dictionary.

    Returns:
        torch.FloatTensor | dict: Transformed input tensor or dictionary.
    """
    means = parameter_dict["means"].clone().detach()  # 1024x3
    quats = parameter_dict["quats"].clone().detach()  # 1024x4
    scales = parameter_dict["scales"].clone().detach()  # 1024x3
    opacities = parameter_dict["opacities"].clone().detach()  # 1024x1
    colors = parameter_dict["colors"].clone().detach()  # 1024x4x3
    # Ks = parameter_dict["Ks"].clone().detach()  # 3x3
    # viewmats = parameter_dict["viewmats"].clone().detach()  # 4x4

    if join_mode == "flatten":
        means = means.view(-1)
        quats = quats.view(-1)
        scales = scales.view(-1)
        opacities = opacities.view(-1)
        colors = colors.view(-1)

        # Concatenate all parameters into a single 1D tensor
        autoencoder_input = torch.cat([means, quats, scales, opacities, colors], dim=0)

        # Normalize the input to the range [-1, 1]
        autoencoder_input = normalize_to_neg_one_one(
            autoencoder_input, autoencoder_input.min(), autoencoder_input.max()
        )
    elif join_mode == "concat":
        means = means.view(32, 32, -1)
        quats = quats.view(32, 32, -1)
        scales = scales.view(32, 32, -1)
        opacities = opacities.view(32, 32, -1)
        colors = colors.view(32, 32, -1)

        # Concatenate all parameters along the channel
        autoencoder_input = torch.cat([means, quats, scales, opacities, colors], dim=2)

        # Normalize the input to the range [-1, 1] along each channel
        for i in range(autoencoder_input.size(2)):
            autoencoder_input[:, :, i] = normalize_to_neg_one_one(
                autoencoder_input[:, :, i],
                autoencoder_input[:, :, i].min(),
                autoencoder_input[:, :, i].max(),
            )

        # Permute values from [32, 32, N] to [N, 32, 32]
        autoencoder_input = autoencoder_input.permute(2, 0, 1)
    elif join_mode == "dict":
        parameter_dict = {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
        }

        # For each parameter, convert to 32x32 and normalize to [-1, 1]
        for key, value in parameter_dict.items():
            # Convert to 32x32
            value = value.view(32, 32, -1)

            # Normalize the input to the range [-1, 1] along each channel
            for i in range(value.size(2)):
                value[:, :, i] = normalize_to_neg_one_one(
                    value[:, :, i],
                    value[:, :, i].min(),
                    value[:, :, i].max(),
                )

            # Permute values from [32, 32, N] to [N, 32, 32]
            value = value.permute(2, 0, 1)

            parameter_dict[key] = value

        autoencoder_input = parameter_dict
    else:
        raise ValueError(f"Invalid join_mode: {join_mode}")

    return autoencoder_input


def transform_autoencoder_output(
    autoencoder_output: torch.Tensor | Dict[str, torch.Tensor],
    join_mode: str = "flatten",
) -> Dict[str, torch.Tensor]:
    """
    Reconstructs the Gaussian splat parameter dictionary from the autoencoder output.

    Args:
        autoencoder_output (torch.FloatTensor): Flattened tensor from the autoencoder output.
        join_mode (str): Determines how to concatenate tensors:
            - "flatten": Flatten all tensors and concatenate them, result is a 1D tensor.
            - "concat": Concatenate all tensors along the channel, result is a 2D tensor (32x32xN).
            - "dict": Converts to 2D tensor and returns as a dictionary.

    Returns:
        dict: Reconstructed parameter dictionary.
    """
    if join_mode == "flatten":
        # Denormalize the output to the original range
        autoencoder_output = denormalize_from_neg_one_one(
            autoencoder_output, autoencoder_output.min(), autoencoder_output.max()
        )

        # Reconstruct each parameter from the flattened tensor
        idx = 0

        means_size = 1024 * 3
        means = (
            autoencoder_output[idx : idx + means_size].clone().detach().view(1024, 3)
        )
        idx += means_size

        quats_size = 1024 * 4
        quats = (
            autoencoder_output[idx : idx + quats_size].clone().detach().view(1024, 4)
        )
        idx += quats_size

        scales_size = 1024 * 3
        scales = (
            autoencoder_output[idx : idx + scales_size].clone().detach().view(1024, 3)
        )
        idx += scales_size

        opacities_size = 1024
        opacities = (
            autoencoder_output[idx : idx + opacities_size].clone().detach().view(1024)
        )
        idx += opacities_size

        colors_size = 1024 * 4 * 3
        colors = (
            autoencoder_output[idx : idx + colors_size]
            .clone()
            .detach()
            .view(1024, 4, 3)
        )

    elif join_mode == "concat":
        # Permute values from [N, 32, 32] to [32, 32, N]
        autoencoder_output = autoencoder_output.permute(1, 2, 0)

        # Denormalize the output to the original range along each channel
        for i in range(autoencoder_output.size(2)):
            autoencoder_output[:, :, i] = denormalize_from_neg_one_one(
                autoencoder_output[:, :, i],
                autoencoder_output[:, :, i].min(),
                autoencoder_output[:, :, i].max(),
            )

        # Reconstruct each parameter from the concatenated tensor
        idx = 0

        means_size = 3
        means = (
            autoencoder_output[:, :, idx : idx + means_size]
            .clone()
            .detach()
            .view(1024, 3)
        )
        idx += means_size

        quats_size = 4
        quats = (
            autoencoder_output[:, :, idx : idx + quats_size]
            .clone()
            .detach()
            .view(1024, 4)
        )
        idx += quats_size

        scales_size = 3
        scales = (
            autoencoder_output[:, :, idx : idx + scales_size]
            .clone()
            .detach()
            .view(1024, 3)
        )
        idx += scales_size

        opacities_size = 1
        opacities = (
            autoencoder_output[:, :, idx : idx + opacities_size]
            .clone()
            .detach()
            .view(1024)
        )
        idx += opacities_size

        colors_size = 4 * 3
        colors = (
            autoencoder_output[:, :, idx : idx + colors_size]
            .clone()
            .detach()
            .view(1024, 4, 3)
        )
    elif join_mode == "dict":
        # Denormalize the output to the original range along each channel
        for _, value in autoencoder_output.items():
            # Permute values from [N, 32, 32] to [32, 32, N]
            value = value.permute(1, 2, 0)
            
            for i in range(value.size(2)):
                value = value.clone().detach()
                value[:, :, i] = denormalize_from_neg_one_one(
                    value[:, :, i], value[:, :, i].min(), value[:, :, i].max()
                )

        # Reconstruct each parameter from the concatenated tensor
        means = autoencoder_output["means"].view(1024, 3)
        quats = autoencoder_output["quats"].view(1024, 4)
        scales = autoencoder_output["scales"].view(1024, 3)
        opacities = autoencoder_output["opacities"].view(1024)
        colors = autoencoder_output["colors"].view(1024, 4, 3)
    else:
        raise ValueError(f"Invalid join_mode: {join_mode}")

    # Reconstruct the parameter dictionary
    parameter_dict = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "Ks": CIFAR10_KS,
        "viewmats": CIFAR10_VIEWMATS,
    }

    # Convert to nn.ParameterDict
    parameter_dict = torch.nn.ParameterDict(
        {key: torch.nn.Parameter(value) for key, value in parameter_dict.items()}
    )
    return parameter_dict


def noop_collate(batch: List[Tuple[torch.Tensor, int, Dict[str, Any]]]) -> Any:
    """
    No-op collate function that returns the batch as is.

    Args:
        batch (List[Tuple[torch.Tensor, int, Dict[str, Any]]]): A batch of data tuples.

    Returns:
        Any: The batch as is.
    """
    return batch


def transform_and_collate(
    batch: List[Tuple[torch.Tensor, int, Dict[str, Any]]],
    join_mode: str = "flatten",
    splat_param: str = "means",
) -> torch.Tensor:
    """
    Transforms and collates a batch of data for the autoencoder model.

    Args:
        batch (List[Tuple[torch.Tensor, int, Dict[str, Any]]]): A batch of data tuples.
        join_mode (str): Determines how to concatenate tensors:
            - "flatten": Flatten all tensors and concatenate them, result is a 1D tensor.
            - "concat": Concatenate all tensors along the channel, result is a 2D tensor (32x32xN).
                        Note that if batch = 1, model will learn features separately, otherwise combined.
            - "dict": Converts to 2D tensor and returns as a dictionary.
        splat_param (str) : The parameter to be used for the splatting operation.

    Returns:
        torch.Tensor: The transformed and collated batch.
    """
    # Extract and transform the last element of each tuple
    transformed_data = [
        transform_autoencoder_input(item[-1], join_mode) for item in batch
    ]

    if join_mode == "dict":
        for i in range(len(transformed_data)):
            if splat_param not in transformed_data[i].keys():
                raise ValueError(f"Invalid splat_param: {splat_param}")
            transformed_data[i] = transformed_data[i][splat_param]

    # If only one element in the batch, unsqueeze the tensor along the channel dimension
    if len(transformed_data) == 1 and join_mode == "concat":
        temp_data = transformed_data[0]
        transformed_data = [temp_data[i].unsqueeze(0) for i in range(len(temp_data))]

    return torch.utils.data.dataloader.default_collate(transformed_data)
