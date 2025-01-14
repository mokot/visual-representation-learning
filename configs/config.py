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

    # Results and logs
    save_results: Optional[bool] = False
    results_path: Optional[Path] = Path("./results_grid_init_train")
    save_logs: Optional[bool] = False
    logs_path: Optional[Path] = Path("./logs_grid_init_train")

    # Training settings
    max_steps: int = 1_000
    learning_rate: Optional[float] = 0.01  # Only for group optimization
    loss_weights: List[float] = field(
        default_factory=lambda: [1 / 2, 0, 1 / 2]
    )  # [L1, L2, SSIM]
    normal_loss_weight: Optional[float] = 0.14  # Not for 3DGS, new default value from latest hyperparameter tuning
    distortion_loss_weight: Optional[float] = 0.13  # Not for 3DGS, new default value from latest hyperparameter tuning

    # Regularization settings
    scale_regularization: Optional[float] = 0.1  # New default value from latest hyperparameter tuning
    opacity_regularization: Optional[float] = 0.5  # New default value from latest hyperparameter tuning

    # Gaussian initialization and parameters
    init_type: Literal["random", "grid", "knn"] = (
        "grid"  # `means` are not learnable if `grid`
    )
    num_points: int = 1_024
    extent: float = 2.0  # New default value from latest hyperparameter tuning
    init_scale: float = 2.0  # New default value from latest hyperparameter tuning
    init_opacity: float = 0.6  # New default value from latest hyperparameter tuning

    # Model configuration
    model_type: Literal["2dgs", "2dgs-inria", "3dgs"] = "2dgs"
    sh_degree: Optional[int] = 1  # New default value from latest hyperparameter tuning

    # Optimization settings
    group_optimization: bool = True  # If `True`, strategy is ignored
    strategy: Optional[Literal["default", "mcmc"]] = None
    selective_adam: bool = False  # Only for 3DGS
    sparse_gradient: bool = False

    # Bilateral grid settings
    bilateral_grid: bool = False  # Only for 3DGS

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
