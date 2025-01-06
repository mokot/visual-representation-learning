import math
import time
import torch
import gsplat
import numpy as np
from pathlib import Path
from configs import get_progress_bar, Config
from torch.utils.tensorboard import SummaryWriter
from references import slice, total_variation_loss, BilateralGrid
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from gsplat import (
    rasterization,
    rasterization_2dgs,
    rasterization_2dgs_inria_wrapper,
    DefaultStrategy,
    MCMCStrategy,
)
from utils import (
    append_log,
    compute_knn_distances,
    convert_rgb_to_sh,
    save_gif,
    set_random_seed,
    save_splat,
    save_splat_hdf5,
    save_tensor,
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
        if not cfg.group_optimization:
            if cfg.strategy == "default":
                self.strategy = DefaultStrategy(
                    refine_start_iter=250,
                    refine_every=50,
                    key_for_gradient=(
                        "means2d" if cfg.model_type == "2dgs" else "gradient_2dgs"
                    ),
                )
            elif cfg.strategy == "mcmc":
                self.strategy = MCMCStrategy(
                    refine_start_iter=10,
                    refine_every=10,
                    min_opacity=0.001,
                )
            self.strategy.check_sanity(self.splats, self.optimizers)
            self.strategy_state = self.strategy.initialize_state()

        # Option: Add camera pose optimization (pose_opt)
        # Option: Add appearance optimization (app_opt)

        if cfg.bilateral_grid:
            self.bilateral_grids = BilateralGrid(
                len(self.trainset),
                grid_X=4,
                grid_Y=4,
                grid_W=2,
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bilateral_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Loss and metrics functions
        self.l1 = torch.nn.L1Loss()
        self.mse = torch.nn.MSELoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

    def init_gaussians(self) -> None:
        """
        Initializes Gaussian splats based on the initialization strategy.
        """
        cfg = self.cfg
        if self.init_type == "random":
            # Random initialization
            means = cfg.extent * (
                torch.rand(self.num_points, 3, device=self.device) - 0.5
            )
            quats = torch.rand(self.num_points, 4, device=self.device)
            scales = torch.rand(self.num_points, 3, device=self.device) * cfg.init_scale
            opacities = (
                torch.ones(self.num_points, device=self.device) * cfg.init_opacity
            )

        elif self.init_type == "grid":
            # Grid initialization
            square_root = int(math.sqrt(self.num_points))
            if square_root * square_root != self.num_points:
                raise ValueError("num_points must be a perfect square.")

            cfg.learnable_params["means"] = False

            grid_size = square_root
            bd_min, bd_max = (-1.0, 1.0)

            grid_x = torch.linspace(bd_min, bd_max, grid_size, device=self.device)
            grid_y = torch.linspace(bd_min, bd_max, grid_size, device=self.device)
            grid_z = torch.zeros(
                (self.num_points, 1), device=self.device
            )  # 2D, z is fixed to 0

            mesh = torch.meshgrid(grid_x, grid_y, indexing="ij")
            grid_points_2d = torch.stack(mesh, dim=-1).reshape(-1, 2)
            means = torch.cat([grid_points_2d, grid_z], dim=1)
            quats = torch.rand(self.num_points, 4, device=self.device)
            scales = torch.rand(self.num_points, 3, device=self.device) * cfg.init_scale
            opacities = (
                torch.ones(self.num_points, device=self.device) * cfg.init_opacity
            )

        elif self.init_type == "knn":
            # KNN-based initialization
            means = cfg.extent * (
                torch.rand(self.num_points, 3, device=self.device) * 2 - 1
            )

            # Predefined rotations
            u = torch.rand(self.num_points, 1, device=self.device)
            v = torch.rand(self.num_points, 1, device=self.device)
            w = torch.rand(self.num_points, 1, device=self.device)
            quats = torch.cat(
                [
                    torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                    torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                    torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                    torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
                ],
                -1,
            )

            # Initialize the GS size to be the average dist of the 3 nearest neighbors
            dist2_avg = (compute_knn_distances(means, 4)[:, 1:] ** 2).mean(dim=-1)
            dist_avg = torch.sqrt(dist2_avg)
            scales = (
                torch.log(dist_avg * cfg.init_scale)  # @Rok deleted self.device attribute
                .unsqueeze(-1)
                .repeat(1, 3)
            )

            opacities = torch.logit(
                torch.full((self.num_points,), cfg.init_opacity)  # @Rok deleted self.device attribute
            )

        else:
            raise ValueError(f"Unsupported initialization type: {self.init_type}")

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

        # Parameters (name, values, and learning rate)
        params = [
            (
                "means",
                torch.nn.Parameter(
                    means,
                    requires_grad=(
                        False
                        if self.init_type == "grid"
                        else cfg.learnable_params["means"]
                    ),
                ),
                1.6e-4,
            ),
            (
                "quats",
                torch.nn.Parameter(quats, requires_grad=cfg.learnable_params["quats"]),
                1e-3,
            ),
            (
                "scales",
                torch.nn.Parameter(
                    scales, requires_grad=cfg.learnable_params["scales"]
                ),
                5e-3,
            ),
            (
                "opacities",
                torch.nn.Parameter(
                    opacities, requires_grad=cfg.learnable_params["opacities"]
                ),
                5e-2,
            ),
            (
                "colors",
                torch.nn.Parameter(
                    colors, requires_grad=cfg.learnable_params["colors"]
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

        # Color is spherical harmonics coefficients
        if cfg.sh_degree:
            params = [param for param in params if param[0] != "colors"]
            rgbs = torch.rand(self.num_points, 3, device=self.device)
            colors = torch.zeros((self.num_points, (cfg.s + 1) ** 2, 3))
            colors[:, 0, :] = convert_rgb_to_sh(rgbs)
            params.append(
                (
                    "sh0",
                    torch.nn.Parameter(
                        colors[:, :1, :],
                        requires_grad=cfg.learnable_params["colors"],
                    ),
                    2.5e-3,
                )
            )
            params.append(
                (
                    "shN",
                    torch.nn.Parameter(
                        colors[:, 1:, :],
                        requires_grad=cfg.learnable_params["colors"],
                    ),
                    2.5e-3 / 20,
                )
            )

        self.splats = torch.nn.ParameterDict(
            # {name: value for name, value, _ in params if value.requires_grad}  # @Rok changed this as there is a bug when "means" is fixed and it's not saved
            {name: value for name, value, _ in params}
        ).to(self.device)
        self.splat_features = torch.nn.ParameterDict(
            {name: value for name, value, _ in params if not value.requires_grad}
        ).to(self.device)

        if cfg.sparse_gradient:
            optimizer = torch.optim.SparseAdam
        elif cfg.selective_adam:
            optimizer = gsplat.optimizers.SelectiveAdam
        else:
            optimizer = torch.optim.Adam

        if cfg.group_optimization:
            self.optimizers = {
                "optimizer": optimizer(
                    [values for _, values, _ in params if values.requires_grad],
                    cfg.learning_rate,
                )
            }
            # self.optimizers = optimizer(
            #         [values for _, values, _ in params if values.requires_grad],
            #         cfg.learning_rate,
            #     )
            self.schedulers = []
        else:
            self.optimizers = {
                name: optimizer(
                    params=[self.splats[name]],
                    lr=lr * math.sqrt(cfg.batch_size),
                    eps=1e-15 / math.sqrt(cfg.batch_size),
                    betas=(
                        1 - cfg.batch_size * (1 - 0.9),
                        1 - cfg.batch_size * (1 - 0.999),
                    ),
                )
                for name, values, lr in params
                if values.requires_grad
            }
            self.schedulers = [
                torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps)
                )
            ]  # Optimize learning rate for means only

        if cfg.bilateral_grid:
            self.schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            4,
                            start_factor=0.01,
                            total_iters=100,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            4, gamma=0.01 ** (1.0 / cfg.max_steps)
                        ),
                    ]
                )
            )

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
        progress_bar = get_progress_bar(
            range(cfg.max_steps),
            max_steps=cfg.max_steps,
            description="Training Progress",
        )
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
            if cfg.sh_degree:
                colors = torch.cat(
                    [
                        self.splats["sh0"],
                        self.splats["shN"],
                    ],
                    dim=1,
                )
            else:
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
                (
                    render_colors,
                    render_alphas,
                    render_normals,
                    surf_normals,
                    render_distort,
                    _,
                    info,
                ) = rasterization_2dgs(
                    # @Rok forced float to avoid error
                    means=means.float(),
                    quats=quats.float(),
                    scales=scales.float(),
                    opacities=opacities.float(),
                    colors=colors.float(),
                    viewmats=viewmats.float().clone(),
                    Ks=Ks.float(),
                    width=self.W,
                    height=self.H,
                    distloss=cfg.distortion_loss_weight,
                    sparse_grad=cfg.sparse_gradient,
                    packed=False or cfg.sparse_gradient,
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
                    distloss=cfg.distortion_loss_weight,
                    sparse_grad=cfg.sparse_gradient,
                    packed=False or cfg.sparse_gradient,
                )
                render_colors, render_alphas = renders
                render_normals = info["normals_rend"]
                surf_normals = info["normals_surf"]
                render_distort = info["render_distloss"]
                _ = render_colors[..., 3]
            elif self.model_type == "3dgs":
                render_colors, _, info = rasterization(
                    # @Rok forced dtypes to avoid error
                    means=means.float(),
                    quats=quats.float(),
                    scales=scales.float(),
                    opacities=opacities.float(),
                    colors=colors.float(),
                    viewmats=viewmats.float(),
                    Ks=Ks.float(),
                    width=self.W,
                    height=self.H,
                    sparse_grad=cfg.sparse_gradient,
                    packed=False or cfg.sparse_gradient,
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            torch.cuda.synchronize()
            times[0] += time.time() - start

            render_colors = render_colors[0]
            if render_colors.shape[-1] == 4:
                render_colors = render_colors[..., :3]

            if cfg.bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(self.H, device=self.device) + 0.5) / self.H,
                    (torch.arange(self.W, device=self.device) + 0.5) / self.W,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bilateral_grids,
                    grid_xy,
                    colors,
                    torch.zeros((self.H, self.W, 1)),
                )["rgb"]

            if not cfg.group_optimization and cfg.strategy == "default":
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
                render_colors.permute(2, 0, 1).unsqueeze(0),
                image.permute(2, 0, 1).unsqueeze(0),
            )  # BxCxHxW

            # Compute total loss
            loss = l1_loss * cfg.loss_weights[0] + mse_loss * cfg.loss_weights[1] + ssim_loss * cfg.loss_weights[2] # @ Rok this was throwing many errors, so I went for the simpler variant.
            # loss = np.dot([l1_loss, mse_loss, ssim_loss], cfg.loss_weights)  # @Rok changed to torch.dot

            # Option: Add depth loss

            if cfg.normal_loss_weight and self.model_type != "3dgs":
                loss_weight = (
                    cfg.normal_loss_weight if step > cfg.max_steps // 2 else 0.0
                )
                normals = render_normals.squeeze(0).permute((2, 0, 1))
                surf_normals *= render_alphas.squeeze(0).detach()
                if len(surf_normals.shape) == 4:
                    surf_normals = surf_normals.squeeze(0)
                surf_normals = surf_normals.permute((2, 0, 1))
                normal_error = (1 - (normals * surf_normals).sum(dim=0))[None]
                normal_loss = loss_weight * normal_error.mean()
                loss += normal_loss

            if cfg.distortion_loss_weight and self.model_type != "3dgs":
                loss_weight = (
                    cfg.distortion_loss_weight if step > cfg.max_steps // 2 else 0.0
                )
                distortion_loss = render_distort.mean()
                loss += loss_weight * distortion_loss

            if cfg.bilateral_grid:
                tv_loss = 10 * total_variation_loss(self.bilateral_grids.grids)
                loss += tv_loss

            if cfg.scale_regularization:
                loss = (
                    loss
                    + cfg.scale_regularization
                    * torch.abs(torch.exp(self.splats["scales"])).mean()
                )
            if cfg.opacity_regularization:
                loss = (
                    loss
                    + cfg.opacity_regularization
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )

            # Backward step
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start

            # Turn gradients into sparse gradients
            if cfg.sparse_gradient:
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],
                        values=grad[gaussian_ids],
                        size=self.splats[k].size(),
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.selective_adam:
                if cfg.sparse_gradient:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)

            # Optimize the parameters and update the learning rate

            
            # @ Rok new version, I guess this was the intended idea:
            if cfg.selective_adam:
                for optimizer in self.optimizers.values():
                    optimizer.step(visibility_mask)
            else:
                for optimizer in self.optimizers.values():
                    optimizer.step()
            
            # @ Rok old version which was problematic, as visibility mask is only defined when selective_adam is true:
            # for optimizer in self.optimizers.values():
            #     (   
            #         optimizer.step()
            #         if cfg.selective_adam
            #         else optimizer.step(visibility_mask)
            #     )
                
            for scheduler in self.schedulers:
                scheduler.step()

            if not cfg.group_optimization:
                common_args = {
                    "params": self.splats,
                    "optimizers": self.optimizers,
                    "state": self.strategy_state,
                    "step": step,
                    "info": info,
                }

                if cfg.strategy == "default":
                    self.strategy.step_post_backward(
                        **common_args, packed=False or cfg.sparse_gradient
                    )
                elif cfg.strategy == "mcmc":
                    lr = self.optimizers["means"].param_groups[0]["lr"]
                    self.strategy.step_post_backward(**common_args, lr=lr)

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
                if cfg.normal_loss_weight:
                    self.writer.add_scalar("Loss/Normal", normal_loss.item(), step)
                if cfg.distortion_loss_weight:
                    self.writer.add_scalar(
                        "Loss/Distortion", distortion_loss.item(), step
                    )
                if cfg.bilateral_grid:
                    self.writer.add_scalar("Loss/TV", tv_loss.item(), step)
                # print(self.optimizers["optimizer"].keys())

                # @Rok added clause to avoid bug due to different form of optimizer
                # TODO: could add writer for the case of group_optimization = True
                if not cfg.group_optimization:
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

                # @ Rok commented out as this was having issues
                # save_splat_hdf5(self.splats, self.results_path / f"splat_{step:05d}.h5")

            # Option: Early stopping (based on validation)

        # Save the final results
        save_gif(frames, self.results_path / "animation.gif")
        save_tensor(render_colors, self.results_path / "final.png")
        print(f"Final loss: {loss.item()}")
        print(f"Total Time: Rasterization: {times[0]:.3f}s, Backward: {times[1]:.3f}s")
        save_splat(self.splats, self.results_path / "splat_final.pt")

        # @ Rok commented out as this was having issues
        # save_splat_hdf5(self.splats, self.results_path / "splat_final.h5")

        return render_colors


# TODO: create eval method
