# Note: This is a hack to allow importing from the parent directory
import sys
from pathlib import Path

sys.path.append(str(Path().resolve().parent))

import math
import time
import torch
import numpy as np
import torchmetrics
from torch import optim
from pathlib import Path
from configs import Config
from typing import Optional, Literal
from torch.utils.tensorboard import SummaryWriter
from gsplat import rasterization_2dgs, rasterization_2dgs_inria_wrapper, DefaultStrategy
from utils import load_cifar10, save_gif, save_tensor, set_random_seed


class GaussianImageTrainer:
    """Trains, optimizes and evaluates Gaussian splats to fit a ground truth image."""

    def __init__(self, cfg: Config) -> None:
        """
        Initializes the GaussianImageTrainer.

        Args:
        - cfg (Config): The configuration object.
        """
        # Check if training is possible
        if not torch.cuda.is_available():
            raise Exception("No GPU available. `gpsplat` requires a GPU to train.")

        # Set seed for reproducibility
        set_random_seed(cfg.seed)

        self.cfg = cfg
        self.device = torch.device("cuda")

        # Setup output directories
        self.results_path = Path(cfg.results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.logs_path = Path(cfg.logs_path)
        self.logs_path.mkdir(parents=True, exist_ok=True)

        # Tensorboard logging
        self.writer = SummaryWriter(log_dir=self.logs_path)

        # Load the dataset
        self.train_loader = load_cifar10(
            batch_size=cfg.batch_size,
            shuffle=True,
            train=True,
            data_root=cfg.data_path,
        )
        self.test_loader = load_cifar10(
            batch_size=1,
            shuffle=False,
            train=False,
            data_root=cfg.data_path,
        )

        # Define initialization and model type
        self.init_type = cfg.init_type
        self.model_type = cfg.model_type

        # Camera and image properties
        self.num_points = cfg.num_points
        self.H, self.W = cfg.height, cfg.width
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * math.pi / 2.0)

        # Initialize Gaussian splats
        self.init_gaussians()
        print("Model initialized. Number of Gaussians:", self.num_points)

        # Densification strategy # TODO
        self.strategy = DefaultStrategy(
            verbose=True,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            reset_every=cfg.reset_every,
            refine_every=cfg.refine_every,
            absgrad=cfg.absgrad,
            revised_opacity=cfg.revised_opacity,
            key_for_gradient="gradient_2dgs",
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

        # TODO: Add camera pose optimization
        # TODO: Add appearance optimization

        # Loss and metrics functions
        self.mse = torch.nn.MSELoss()
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(self.device)
        self.psnr = torchmetrics.image.PeakSignalToNoiseRatio(data_range=1.0).to(
            self.device
        )
        self.lpips = torchmetrics.image.lpip.LeastPerceptibleImagePatchSimilarity(
            normalize=True
        ).to(self.device)

    def init_gaussians(self) -> None:
        # TODO: Add support for sparse grad and appearance optimization
        """
        Initializes Gaussian splats based on the initialization strategy.
        """
        if self.init_type == "random":
            # Random initialization
            means = self.cfg.extent * (
                torch.rand(self.num_points, 3, device=self.device) - 0.5
            )
            quats = torch.rand(self.num_points, 4, device=self.device)
            quats = quats / quats.norm(dim=-1, keepdim=True)
            scales = (
                torch.rand(self.num_points, 3, device=self.device) * self.cfg.init_scale
            )
            opacities = (
                torch.ones(self.num_points, device=self.device) * self.cfg.init_opacity
            )
            colors = torch.rand(self.num_points, 3, device=self.device)

        elif self.init_type == "grid":
            # Grid initialization # TODO - check Frederico's code
            pass

        elif self.init_type == "knn":
            # KNN-based initialization # TODO - check source code (references)
            pass
        else:
            raise ValueError(f"Unsupported initialization type: {self.init_type}")

        # Learnable parameters (name, values, and learning rate)
        params = [
            (
                "means",
                torch.nn.Parameter(
                    means, requires_grad=self.cfg.learnable_params["means"]
                ),
                1.6e-4,
            ),
            (
                "quats",
                torch.nn.Parameter(
                    quats, requires_grad=self.cfg.learnable_params["quats"]
                ),
                1e-3,
            ),
            (
                "scales",
                torch.nn.Parameter(
                    scales, requires_grad=self.cfg.learnable_params["scales"]
                ),
                5e-3,
            ),
            (
                "opacities",
                torch.nn.Parameter(
                    opacities, requires_grad=self.cfg.learnable_params["opacities"]
                ),
                5e-2,
            ),
            (
                "colors",
                torch.nn.Parameter(
                    colors, requires_grad=self.cfg.learnable_params["colors"]
                ),
                2.5e-3,
            ),
        ]

        self.splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(
            self.device
        )
        self.optimizers = {
            name: torch.optim.Adam(
                [
                    {
                        "params": [self.splats[name]],
                        "lr": lr * math.sqrt(self.cfg.batch_size),
                    }
                ],
                eps=1e-15 / math.sqrt(self.cfg.batch_size),
                betas=(
                    1 - self.cfg.batch_size * (1 - 0.9),
                    1 - self.cfg.batch_size * (1 - 0.999),
                ),
            )
            for name, _, lr in params
        }

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        results_path: Optional[Path] = None,
        model_type: Literal["2dgs", "3dgs"] = "2dgs",
    ) -> None:
        """
        Trains the random Gaussians to fit the ground truth image.

        Args:
        - iterations (int): Number of iterations to train.
        - lr (float): Learning rate.
        - results_path (Optional[Path]): The path to save the results.
        - model_type (Literal["2dgs", "3dgs"]): Model type ("3dgs" or "2dgs").

        Raises:
        - ValueError: If an unsupported model type is provided.
        """
        # Validate model type
        if model_type not in ["2dgs", "3dgs"]:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Select the rasterization function
        rasterize_fnc = rasterization_2dgs

        # Initialize optimizer and loss
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()

        frames = []
        times = [0, 0]  # (rasterization, backward pass)

        # Camera intrinsics
        K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )

        for iter in range(iterations):
            start = time.time()
            renders = rasterize_fnc(
                self.means,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.scales,
                torch.sigmoid(self.opacities),
                torch.sigmoid(self.rgbs),
                self.viewmat[None],
                K[None],
                self.W,
                self.H,
                packed=False,
            )[0]
            out_image = renders[0]
            torch.cuda.synchronize()
            times[0] += time.time() - start

            # Compute loss and update
            loss = mse_loss(out_image, self.gt_image)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start
            optimizer.step()

            if iter % 100 == 0:
                print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if results_path and iter % 5 == 0:
                frames.append((out_image.detach().cpu().numpy() * 255).astype(np.uint8))

        if results_path:
            save_gif(frames, results_path.with_name(f"animation_{results_path.name}"))
            save_tensor(
                self.gt_image, results_path.with_name(f"original_{results_path.name}")
            )
            save_tensor(out_image, results_path.with_name(f"final_{results_path.name}"))

        print(f"Final loss: {loss.item()}")
        print(f"Total Time: Rasterization: {times[0]:.3f}s, Backward: {times[1]:.3f}s")
