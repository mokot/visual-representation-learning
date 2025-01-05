GRID_SEARCH_SPACE_TEST = {
    # Training settings
    "max_steps": [250, 500, 1000, 2500, 5000, 10000],
    "learning_rate": [None, 1e-4, 1e-3, 1e-2, 1e-1],
    "loss_weights": [
        [1 / 3, 1 / 3, 1 / 3],
        [1 / 2, 1 / 4, 1 / 4],
        [1 / 4, 1 / 2, 1 / 4],
        [1 / 4, 1 / 4, 1 / 2],
    ],
    "normal_loss_weight": [None, 0.1, 0.5, 1.0],
    "distortion_loss_weight": [None, 0.1, 0.5, 1.0],
    # Regularization settings
    "scale_regulation": [None, 0.1, 0.5, 1.0],
    "opacity_regulation": [None, 0.1, 0.5, 1.0],
    # Gaussian initialization and parameters
    "init_type": ["random", "grid", "knn"],
    "num_points": [256, 512, 1024, 2048],
    "extent": [1.0, 2.0, 4.0],
    "init_scale": [0.5, 1.0, 2.0],
    "init_opacity": [0.5, 1.0, 2.0],
    # Model configuration
    "model_type": ["2dgs", "2dgs-inria", "3dgs"],
    "sh_degree": [None, 3],
    # Optimization settings
    "group_optimization": [True, False],
    "strategy": [None, "default", "mcmc"],
    "selective_adam": [True, False],
    "sparse_gradient": [True, False],
    # Bilateral grid settings
    "bilateral_grid": [True, False],
}


GRID_SEARCH_SPACE_TEST_0 = {
    "max_steps": [500, 1000],
    "num_points": [16 * 16, 32 * 32],
    "init_type": ["random", "grid"],
}

GRID_SEARCH_SPACE_TEST_1 = {
    "number_iterations": [500, 1000, 2500, 5000, 10000],
    "number_ipo": [1024],
}

GRID_SEARCH_SPACE_TEST_2 = {
    "number_iterations": [1000],
    "eps2d": [2, 4, 8, 16],
}
