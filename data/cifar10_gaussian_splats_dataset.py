import os
import torch
import random
from pathlib import Path
from utils import load_gs_data
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, Optional, Tuple


class CIFAR10GaussianSplatsDataset(Dataset):

    def __init__(
        self,
        root: str,
        train: bool = True,
        init_type: str = "grid",  # grid or knn
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root (str): Base directory containing all folders for different init types.
            train (bool): If True, loads training split. If False, loads test split.
            init_type (str): Subdirectory corresponding to the initialization type ['grid' or 'knn'].
            transform (callable, optional): Optional transform to be applied on a PIL image.
        """
        self.root = os.path.join(root, init_type, "train" if train else "test")
        if not os.path.exists(self.root):
            raise ValueError(
                f"The folder for init_type '{init_type}' and split {'train' if train else 'test'} does not exist in {root}."
            )

        self.file_names = [f for f in os.listdir(self.root) if f.endswith(".pt")]
        self.transform = transform

        # Class to index mapping
        self.class_to_index = {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
            "ship": 8,
            "truck": 9,
        }

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = Path(os.path.join(self.root, self.file_names[idx]))

        # Use the provided load_gs_data function
        image, label, params = load_gs_data(file_path)

        if self.transform:
            image = self.transform(image)

        return image, label, params

    def shuffle(self):
        random.shuffle(self.file_names)
