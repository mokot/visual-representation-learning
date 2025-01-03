from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal


@dataclass
class Config:
    # Seed for reproducibility
    seed: int = 42

    # Path for the dataset
    data_path: Optional[Path] = Path("./data")

    # Batch size (for training)
    batch_size: int = 1

    # Paths for saving results and logs
    results_path: Optional[Path] = Path("./results")
    logs_path: Optional[Path] = Path("./logs")

    # Dataset and image settings (CIFAR-10 resolution)
    height: int = 32
    width: int = 32
    normalize: bool = True  # Normalize images

    # Training settings
    max_steps: int = 1_000
    eval_steps: List[int] = field(default_factory=lambda: [250, 500, 750, 1_000])
    save_steps: List[int] = field(default_factory=lambda: [500, 1_000])
    learning_rate: float = 1e-3  # Learning rate (for Adam optimizer)

    # Model type and rasterization
    model_type: Literal["2dgs", "2dgs-inria"] = "2dgs"

    # Gaussian initialization and strategy
    init_type: Literal["random", "grid", "knn"] = "random"
    num_points: int = 1_024  # Number of Gaussians (32x32)
    extent: float = 2.0  # Extent of Gaussians
    init_opacity: float = 0.5  # Initial opacity of Gaussians
    init_scale: float = 1.0  # Initial scale of Gaussians

    # Spherical harmonics
    sh_degree: int = 3
    sh_degree_interval: int = 1000  # Interval for changing SH degree

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
