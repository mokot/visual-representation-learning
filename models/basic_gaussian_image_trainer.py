import math
import time
import torch
import numpy as np
from torch import optim
from pathlib import Path
from utils import save_gif, save_tensor
from typing import Optional, Literal
from gsplat import rasterization, rasterization_2dgs


class GaussianImageTrainer:
    """Trains random Gaussians to fit an image."""

    def __init__(self, gt_image: torch.Tensor, num_points: int = 1024) -> None:
        """
        Initializes the GaussianImageTrainer.

        Args:
        - gt_image (torch.Tensor): The ground truth image to fit.
        - num_points (int): The number of Gaussians to use.

        Raises:
        - ValueError: If gt_image is not a 2D or 3D tensor.
        """
        if not isinstance(gt_image, torch.Tensor):
            raise TypeError("gt_image must be a torch.Tensor.")
        if gt_image.ndim not in [2, 3]:
            raise ValueError("gt_image must be a 2D or 3D tensor.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        # Camera and image properties
        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.image_size = torch.tensor([self.W, self.H, 1], device=self.device)

        # Initialize Gaussian parameters
        self._init_gaussians()

    def _init_gaussians(self) -> None:
        """
        Initializes random Gaussians for the optimization.
        """
        # Bounding box size
        bd = 2

        # Initialize Gaussian properties
        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        self.rgbs = torch.rand(self.num_points, 3, device=self.device)

        # Initialize quaternion rotations
        u, v, w = [torch.rand(self.num_points, 1, device=self.device) for _ in range(3)]
        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.quats = self.quats / self.quats.norm(dim=-1, keepdim=True)

        # Initialize opacities and view matrix
        self.opacities = torch.ones((self.num_points), device=self.device)
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        # Background color
        self.background = torch.zeros(3, device=self.device)

        # Set gradients for trainable parameters
        for param in [
            self.means,
            self.scales,
            self.quats,
            self.rgbs,
            self.opacities,
        ]:
            param.requires_grad = True
        for param in [
            self.viewmat,
            self.background,
        ]:
            param.requires_grad = False

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
        rasterize_fnc = rasterization if model_type == "3dgs" else rasterization_2dgs

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
