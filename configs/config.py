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

    # Image configuration
    image: torch.Tensor = torch.rand(32, 32, 3)
    batch_size: int = 1

    # Paths
    results_path: Optional[Path] = Path("./results")
    logs_path: Optional[Path] = Path("./logs")

    # Training settings
    max_steps: int = 1_000
    learning_rate: Optional[float] = 1e-3
    loss_weights: List[float] = [0.33, 0.33, 0.33]  # [L1, MSE, SSIM]
    normal_loss_weight: Optional[float] = None  # Not for 3DGS
    distortion_loss_weight: Optional[float] = None  # Not for 3DGS

    # Regularization settings
    scale_reg: Optional[float] = None
    opacity_reg: Optional[float] = None

    # Gaussian initialization and parameters
    init_type: Literal["random", "grid", "knn"] = (
        "random"  # `means` are not learnable if `grid`
    )
    num_points: int = 1_024
    extent: float = 2.0
    init_scale: float = 1.0
    init_opacity: float = 1.0

    # Model configuration
    model_type: Literal["2dgs", "2dgs-inria", "3dgs"] = "2dgs"
    sh_degree: Optional[int] = None

    # Optimization settings
    group_optimization: Optional[bool] = True  # If `True`, strategy is ignored
    strategy: Optional[Literal["default", "mcmc"]] = None
    selective_adam: Optional[bool] = False
    sparse_gradient: Optional[bool] = False

    # Bilateral grid settings
    bilateral_grid: Optional[bool] = False

    # Learnable parameters
    learnable_params: dict = field(
        default_factory=lambda: {
            "means": True,
            "quats": True,
            "scales": True,
            "opacities": True,
            "colors": True,
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
