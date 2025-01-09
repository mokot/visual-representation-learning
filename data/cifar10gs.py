import torch
from torch.utils.data import Dataset
import os
import pickle


class CIFAR10GaussianSplatsDataset(Dataset):
    """
    Custom PyTorch Dataset for CIFAR10 images transformed into Gaussian splats.

    Each sample contains:
    - The original CIFAR10 image
    - Its label (target)
    - The corresponding Gaussian splat represented as a ParameterDict
    """

    def __init__(self, data, targets, splats, transform=None):
        """
        Initialize the dataset.

        Args:
            data (torch.Tensor): Tensor of CIFAR10 images.
            targets (torch.Tensor): Tensor of CIFAR10 labels.
            splats (list[dict]): List of Gaussian splats represented as dicts.
            transform (callable, optional): Transformation to apply to images.
        """
        self.data = data
        self.targets = targets
        self.splats = splats
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        splat = self.splats[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, splat

    @staticmethod
    def save_dataset(dataset, path):
        """
        Save the dataset to a file.

        Args:
            dataset (CIFAR10GaussianSplatsDataset): The dataset to save.
            path (str): The file path to save the dataset.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "data": dataset.data,
                    "targets": dataset.targets,
                    "splats": dataset.splats,
                },
                f,
            )

    @staticmethod
    def load_dataset(path):
        """
        Load the dataset from a file.

        Args:
            path (str): The file path to load the dataset from.

        Returns:
            CIFAR10GaussianSplatsDataset: The loaded dataset.
        """
        with open(path, "rb") as f:
            dataset_dict = pickle.load(f)
        return CIFAR10GaussianSplatsDataset(
            data=dataset_dict["data"],
            targets=dataset_dict["targets"],
            splats=dataset_dict["splats"],
        )
