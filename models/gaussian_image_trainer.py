import math
import time
import tqdm
import torch
import numpy as np
import torchmetrics
from pathlib import Path
from configs import Config
from torch.utils.tensorboard import SummaryWriter
from gsplat import (
    rasterization,
    rasterization_2dgs,
    rasterization_2dgs_inria_wrapper,
    MCMCStrategy,
)
from utils import (
    append_log,
    save_gif,
    save_splat,
    save_splat_hdf5,
    save_tensor,
    set_random_seed,
)


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
        self.H, self.W = cfg.image.shape[:2]

        # Initialize Gaussian splats
        self.init_gaussians()
        print("Model initialized. Number of Gaussians:", self.num_points)

        # Densification strategy
        self.strategy = MCMCStrategy(
            refine_start_iter=50,
            refine_every=10,
            min_opacity=0.001,
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
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(
            self.device
        )
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
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

        elif self.init_type == "grid":
            # Option: Grid initialization
            # TODO: means should not be learnable
            square_root = round(math.sqrt(self.num_points))
            if square_root**2 != self.num_points:
                raise ValueError(
                    f"When init_type is grid, num_points must be a perfect square. Received num_points={self.num_points}."
                )

            grid_size = square_root
            bd_min, bd_max = (-1.0, 1.0)

            # 2D Grid Initialization (z = 0)
            grid_x = torch.linspace(bd_min, bd_max, grid_size, device=self.device)
            grid_y = torch.linspace(bd_min, bd_max, grid_size, device=self.device)

            mesh = torch.meshgrid(grid_x, grid_y, indexing="ij")
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

        # Parameters (name, values, and learning rate)
        params = [
            (
                "means",
                torch.nn.Parameter(
                    means,
                    requires_grad=(
                        False
                        if self.init_type == "grid"
                        else self.cfg.learnable_params["means"]
                    ),
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
            (
                "viewmats",
                torch.nn.Parameter(viewmats, requires_grad=False),
                0.0,
            ),
            (
                "Ks",
                torch.nn.Parameter(Ks, requires_grad=False),
                0.0,
            ),
        ]

        # Option: Feature-based dimensionality, where color is spherical harmonics

        self.splats = torch.nn.ParameterDict(
            {name: value for name, value, _ in params if value.requires_grad}
        ).to(self.device)
        self.splat_features = torch.nn.ParameterDict(
            {name: value for name, value, _ in params if not value.requires_grad}
        ).to(self.device)

        self.optimizers = {
            name: torch.optim.Adam(
                params=[self.splats[name]],
                lr=lr * math.sqrt(self.cfg.batch_size),
                eps=1e-15 / math.sqrt(self.cfg.batch_size),
                betas=(
                    1 - self.cfg.batch_size * (1 - 0.9),
                    1 - self.cfg.batch_size * (1 - 0.999),
                ),
            )
            for name, values, lr in params
            if values.requires_grad
        }
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=0.01 ** (1.0 / self.cfg.max_steps)
        )  # Optimize learning rate for means only

    def train(
        self,
    ) -> None:
        """
        Trains the Gaussians to fit the ground truth image.
        """
        cfg = self.cfg
        cfg.save(self.logs_path / "config.json")
        image = self.image
        save_tensor(image, self.results_path / "original.png")

        # Training loop
        frames = []
        times = [0, 0]  # (rasterization, backward pass)
        progress_bar = tqdm.tqdm(range(cfg.max_steps))
        for step in progress_bar:
            means = (
                self.splats["means"]
                if "means" in self.splats
                else self.splat_features["means"]
            )
            quats = (
                self.splats["quats"]
                if "quats" in self.splats
                else self.splat_features["quats"]
            )
            quats = quats / quats.norm(dim=-1, keepdim=True)
            scales = (
                self.splats["scales"]
                if "scales" in self.splats
                else self.splat_features["scales"]
            )
            opacities = torch.sigmoid(
                self.splats["opacities"]
                if "opacities" in self.splats
                else self.splat_features["opacities"]
            )
            colors = torch.sigmoid(
                self.splats["colors"]
                if "colors" in self.splats
                else self.splat_features["colors"]
            )
            viewmats = self.splat_features["viewmats"][None]
            Ks = self.splat_features["Ks"][None]

            start = time.time()

            # Rasterize the splats
            if self.model_type == "2dgs":
                (render_colors, _, _, _, _, _, info) = rasterization_2dgs(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=self.W,
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
                    width=self.W,
                    height=self.H,
                    packed=False,
                )
                render_colors, _ = renders
                _ = info["normals_rend"]
                _ = info["normals_surf"]
                _ = info["render_distloss"]
                _ = render_colors[..., 3]
            elif self.model_type == "3dgs":
                render_colors, _, info = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=self.W,
                    height=self.H,
                    packed=False,
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            torch.cuda.synchronize()
            times[0] += time.time() - start

            render_colors = render_colors[0]
            if render_colors.shape[-1] == 4:
                render_colors = render_colors[..., :3]

            # Compute loss
            l1_loss = self.l1(render_colors, image)
            mse_loss = self.mse(render_colors, image)
            ssim_loss = 1.0 - self.ssim(
                render_colors.permute(2, 0, 1).unsqueeze(0),
                image.permute(2, 0, 1).unsqueeze(0),
            )  # BxCxHxW

            # Option: Add depth loss
            # Option: Add normal loss
            # Option: Add distortion loss

            # Compute total loss
            loss = mse_loss

            # Backward step
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start

            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                lr=self.optimizers["means"].param_groups[0]["lr"],
            )

            # Optimize the parameters and update the learning rate
            for optimizer in self.optimizers.values():
                optimizer.step()
            self.scheduler.step()

            # Save logs and results
            if step % 5 == 0:
                description = f"Loss: {loss:.3f} (L1: {l1_loss:.3f}, MSE: {mse_loss:.3f}, SSIM: {ssim_loss:.3f})"
                progress_bar.set_description(description)
                append_log(description, self.logs_path / "logs.txt")
                frames.append(
                    (render_colors.detach().cpu().numpy() * 255).astype(np.uint8)
                )
            if step % 100 == 0:
                save_tensor(render_colors, self.results_path / f"step_{step:05d}.png")
                # Tensorboard logging
                self.writer.add_scalar(
                    "Number of Gaussians", len(self.splats["means"]), step
                )
                self.writer.add_scalar("Loss/Total", loss.item(), step)
                self.writer.add_scalar("Loss/L1", l1_loss.item(), step)
                self.writer.add_scalar("Loss/MSE", mse_loss.item(), step)
                self.writer.add_scalar("Loss/SSIM", ssim_loss.item(), step)
                self.writer.add_scalar(
                    "LearningRate", self.optimizers["means"].param_groups[0]["lr"], step
                )
                self.writer.add_scalar(
                    "Memory/Allocated", torch.cuda.memory_allocated(), step
                )
                canvas = torch.cat(
                    [
                        render_colors.permute(2, 0, 1).unsqueeze(0),
                        image.permute(2, 0, 1).unsqueeze(0),
                    ],
                    dim=-1,
                )
                canvas = canvas.detach().cpu().numpy()
                self.writer.add_image("Rendered vs. Original", canvas.squeeze(0), step)
                self.writer.flush()
                save_splat(self.splats, self.results_path / f"splat_{step:05d}.pt")
                save_splat_hdf5(self.splats, self.results_path / f"splat_{step:05d}.h5")

        # Save the final results
        save_gif(frames, self.results_path / "animation.gif")
        save_tensor(render_colors, self.results_path / "final.png")
        print(f"Final loss: {loss.item()}")
        print(f"Total Time: Rasterization: {times[0]:.3f}s, Backward: {times[1]:.3f}s")


# TODO: create eval method
