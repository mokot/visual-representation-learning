# Note: This is a hack to allow importing from the parent directory
import sys
from pathlib import Path

sys.path.append(str(Path().resolve().parent))

import json
import math
import time
import tqdm
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
        self.image = cfg.image.to(self.device)

        # Setup output directories
        self.results_path = Path(cfg.results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.logs_path = Path(cfg.logs_path)
        self.logs_path.mkdir(parents=True, exist_ok=True)

        # Tensorboard logging
        self.writer = SummaryWriter(log_dir=self.logs_path)

        # Define initialization and model type
        self.init_type = cfg.init_type
        self.model_type = cfg.model_type

        # Camera and image properties
        self.num_points = cfg.num_points
        self.H, self.W = self.image.shape[-2:]

        # Initialize Gaussian splats
        self.init_gaussians()
        print("Model initialized. Number of Gaussians:", self.num_points)

        # Densification strategy
        self.strategy = DefaultStrategy(
            key_for_gradient="gradient_2dgs",  # Needed for 2dgs
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

        # Option: Add camera pose optimization
        # Option: Add appearance optimization

        # Loss and metrics functions
        self.l1 = torch.nn.L1Loss()
        self.mse = torch.nn.MSELoss()
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(self.device)
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(  # @Rok changed to correct name
            self.device
        )
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(  #  @Rok changed to correct name
            normalize=True
        ).to(self.device)

    def init_gaussians(self) -> None:
        """
        Initializes Gaussian splats based on the initialization strategy.
        """
        if self.init_type == "random":
            # Random initialization
            means = self.cfg.extent * (
                torch.rand(self.num_points, 3, device=self.device) - 0.5
            )
            quats = torch.rand(self.num_points, 4, device=self.device)
            scales = (
                torch.rand(self.num_points, 3, device=self.device) * self.cfg.init_scale
            )
            opacities = (
                torch.ones(self.num_points, device=self.device) * self.cfg.init_opacity
            )
            colors = torch.rand(self.num_points, 3, device=self.device)
            self.viewmats = torch.tensor(  # @Rok Directly set self.viewmats here instead of setting it as a parameter
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 8.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=self.device,
            )
            focal = 0.5 * float(self.W) / math.tan(0.5 * math.pi / 2.0)
            self.Ks = torch.tensor(  # @Rok  Directly set self.Ks here
                [
                    [focal, 0, self.W / 2],
                    [0, focal, self.H / 2],
                    [0, 0, 1],
                ],
                device=self.device,
            )

        elif self.init_type == "grid":
            # Option: Grid initialization
            # TODO: means should not be learnable
            square_root = round(math.sqrt(self.num_points))
            if square_root ** 2 != self.num_points:
                raise ValueError(f"When init_type is grid, num_points must be a perfect square. Received num_points={self.num_points}.")
                
            grid_size = square_root
            bd_min, bd_max = (-1.0, 1.0)
            
            # 2D Grid Initialization (z = 0)
            grid_x = torch.linspace(bd_min, bd_max, grid_size, device=self.device)
            grid_y = torch.linspace(bd_min, bd_max, grid_size, device=self.device)
            
            mesh = torch.meshgrid(grid_x, grid_y, indexing='ij')
            grid_points_2d = torch.stack(mesh, dim=-1).reshape(-1, 2)
            
            # Append a fixed z-coordinate (e.g., z = 0)
            fixed_z = torch.zeros((self.num_points, 1), device=self.device)
            self.means = torch.cat([grid_points_2d, fixed_z], dim=1)

            
            # TODO: Actually, only the means should change according to the init_type to avoid repeating the following block:
            quats = torch.rand(self.num_points, 4, device=self.device)
            scales = (
                torch.rand(self.num_points, 3, device=self.device) * self.cfg.init_scale
            )
            opacities = (
                torch.ones(self.num_points, device=self.device) * self.cfg.init_opacity
            )
            colors = torch.rand(self.num_points, 3, device=self.device)
            viewmats = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 8.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=self.device,
            )
            focal = 0.5 * float(self.W) / math.tan(0.5 * math.pi / 2.0)
            Ks = torch.tensor(
                [
                    [focal, 0, self.W / 2],
                    [0, focal, self.H / 2],
                    [0, 0, 1],
                ],
                device=self.device,
            )

        elif self.init_type == "knn":
            # Option: KNN-based initialization
            pass
        else:
            raise ValueError(f"Unsupported initialization type: {self.init_type}")

        # Learnable parameters (name, values, and learning rate)
        means_req_grad = True
        if self.init_type == "grid":
            means_req_grad = False
        
        params = [
            (
                "means",                
                torch.nn.Parameter(
                    means, requires_grad=means_req_grad  # @Rok I hope this solution is OK for you
                ),
                1.6e-4,  # @Rok what do these numbers mean?
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
            # (  #  @Rok comented out as this lead to a problem with the sanity check.
            #     "viewmats",
            #     torch.nn.Parameter(
            #         viewmats, requires_grad=self.cfg.learnable_params["viewmats"]
            #     ),
            #     0.0,
            # ),
            # (
            #     "Ks",
            #     torch.nn.Parameter(Ks, requires_grad=self.cfg.learnable_params["Ks"]),
            #     0.0,
            # ),
        ]

        # Option: Feature-based dimensionality, where color is spherical harmonics

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
            for name, values, lr in params
            if values.requires_grad
        }

    def train(
        self,
    ) -> None:
        """
        Trains the Gaussians to fit the ground truth image.
        """
        cfg = self.cfg
        cfg.save(self.logs_path / "config.json")  # @Rok changed method
        image = self.image
        save_tensor(image, self.results_path / "original.png")

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps)
        )

        # Training loop
        frames = []
        times = [0, 0]  # (rasterization, backward pass)
        progress_bar = tqdm.tqdm(range(cfg.max_steps))
        for step in progress_bar:
            means = self.splats["means"]
            quats = self.splats["quats"] / self.splats["quats"].norm(
                dim=-1, keepdim=True
            )
            scales = self.splats["scales"]
            opacities = torch.sigmoid(self.splats["opacities"])
            colors = torch.sigmoid(self.splats["colors"])

            # @Rok changed this here to use the attributes defined above as using params caused an error in the sanity check
            # viewmats = self.splats["viewmats"]
            viewmats = self.viewmats
            # Ks = self.splats["Ks"]
            Ks = self.Ks

            start = time.time()

            # Rasterize the splats
            if self.model_type == "2dgs":
                (
                    render_colors,
                    render_alphas,
                    render_normals,
                    normals_from_depth,
                    render_distort,
                    render_median,
                    info,
                ) = rasterization_2dgs(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=self.W,  # @ Rok used correct parameter name
                    height=self.H,
                    packed=False,
                )
            elif self.model_type == "2dgs-inria":
                renders, info = rasterization_2dgs_inria_wrapper(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmats,
                    Ks=Ks,
                    W=self.W,
                    H=self.H,
                    packed=False,
                )
                render_colors, render_alphas = renders
                render_normals = info["normals_rend"]
                normals_from_depth = info["normals_surf"]
                render_distort = info["render_distloss"]
                render_median = render_colors[..., 3]
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            torch.cuda.synchronize()
            times[0] += time.time() - start

            if render_colors.shape[-1] == 4:
                render_colors = render_colors[..., :3]

            # Pre-backward step
            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # Compute loss
            l1_loss = self.l1(render_colors, image)
            mse_loss = self.mse(render_colors, image)
            ssim_loss = 1.0 - self.ssim(
                render_colors.permute(0, 3, 1, 2), image.permute(0, 3, 1, 2)
            )

            # Option: Add depth loss
            # Option: Add normal loss
            # Option: Add distortion loss

            loss = (l1_loss + mse_loss + ssim_loss) / 3.0

            # Backward step
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()

            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start

            # Compute total loss
            loss = np.mean([l1_loss, mse_loss, ssim_loss])
            description = f"Loss: {loss:.3f} (L1: {l1_loss:.3f}, MSE: {mse_loss:.3f}, SSIM: {ssim_loss:.3f})"
            progress_bar.set_description(description)

            # TODO: Tensorboard logging

            # Post-backward step
            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=False,
            )

            # Optimize the parameters and update the learning rate
            for optimizer in self.optimizers.values():
                optimizer.step()
            scheduler.step()

            # Save logs and results
            if step % 5 == 0:
                frames.append(
                    (render_colors.detach().cpu().numpy() * 255).astype(np.uint8)
                )
            if step % 100 == 0:
                save_tensor(render_colors, self.results_path / f"step_{step:05d}.png")
                # Save the current splat into log file - TODO

        # Save the final results
        save_gif(frames, self.results_path / "animation.gif")
        save_tensor(render_colors, self.results_path / "final.png")
        print(f"Final loss: {loss.item()}")
        print(f"Total Time: Rasterization: {times[0]:.3f}s, Backward: {times[1]:.3f}s")

    # def save(self, path: Path) -> None:
    #     """
    #     Saves the object's state to a JSON file.

    #     Args:
    #         path (Path): The file path to save the object's state.
    #     """
    #     with open(path, "w") as f:
    #         json.dump(self.__dict__, f, indent=4, default=custom_serializer)
