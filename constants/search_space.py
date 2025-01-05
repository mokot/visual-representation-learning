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

# Lets first figure out if the strategy is working
# After you are done, update the default values in config.py
GRID_SEARCH_SPACE_TEST_1 = {
    # Optimization settings
    "group_optimization": [True, False],
    "strategy": [None, "default", "mcmc"],
}

# Lets find how are losses affecting the training
GRID_SEARCH_SPACE_TEST_2 = {
    "loss_weights": [
        [1 / 3, 1 / 3, 1 / 3],
        [1 / 2, 1 / 4, 1 / 4],
        [1 / 4, 1 / 2, 1 / 4],
        [1 / 4, 1 / 4, 1 / 2],
    ],
    "normal_loss_weight": [None, 0.1, 0.5, 1.0],
    "distortion_loss_weight": [None, 0.1, 0.5, 1.0],
}

# Lets figure out if the regularization helps
GRID_SEARCH_SPACE_TEST_3 = {
    "scale_regulation": [None, 0.1, 0.5, 1.0],
    "opacity_regulation": [None, 0.1, 0.5, 1.0],
}

# What about initializations?
GRID_SEARCH_SPACE_TEST_4 = {
    "init_type": ["random", "grid", "knn"],
    "num_points": [256, 512, 1024, 2048],
    "extent": [1.0, 2.0, 4.0],
    "init_scale": [0.5, 1.0, 2.0],
    "init_opacity": [0.5, 1.0, 2.0],
}

# Main thingies (you can omit 2dgs-inria since it might have some issues)
GRID_SEARCH_SPACE_TEST_5 = {
    # Training settings
    "max_steps": [250, 500, 1000, 2500, 5000, 10000],
    "learning_rate": [None, 1e-4, 1e-3, 1e-2, 1e-1],
    # Model configuration
    "model_type": ["2dgs", "2dgs-inria", "3dgs"],
}

GRID_SEARCH_SPACE_TEST_4 = {
    "sh_degree": [None, 3],
    "selective_adam": [True, False],
    "sparse_gradient": [True, False],
    "bilateral_grid": [True, False],
}
