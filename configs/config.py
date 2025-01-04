import json
import torch
from pathlib import Path
from utils import custom_serializer
from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class Config:
    # Seed for reproducibility
    seed: int = 42

    # Image
    image: torch.Tensor = torch.rand(3, 32, 32)

    # Batch size (for training)
    batch_size: int = 1

    # Paths for saving results and logs
    results_path: Optional[Path] = Path("./results")
    logs_path: Optional[Path] = Path("./logs")

    # Training settings
    max_steps: int = 1_000
    learning_rate: float = 1e-3  # Learning rate (for Adam optimizer)

    # Model type and rasterization
    model_type: Literal["2dgs", "2dgs-inria"] = "2dgs"

    # Gaussian initialization and strategy
    init_type: Literal["random", "grid", "knn"] = "random"  # TODO @Rok maybe only specify init_means as the rest is redundant?
    num_points: int = 1_024  # Number of Gaussians (32x32)
    extent: float = 2.0  # Extent of Gaussians
    init_opacity: float = 0.5  # Initial opacity of Gaussians
    init_scale: float = 1.0  # Initial scale of Gaussians

    # Learnable parameters
    learnable_params: dict = field(
        default_factory=lambda: {
            "means": True,
            "quats": True,
            "scales": True,
            "opacities": True,
            "colors": True,
            "viewmats": False,
            "Ks": False,
        }
    )


def save(self, path: Path) -> None:
    """
    Saves the object's state to a JSON file.

    Args:
        path (Path): The file path to save the object's state.
    """
    with open(path, "w") as f:
        json.dump(self.__dict__, f, indent=4, default=custom_serializer)
