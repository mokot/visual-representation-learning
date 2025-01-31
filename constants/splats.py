import math
import torch

# Constants for CIFAR-10 dataset
CIFAR_WIDTH = 32
CIFAR_HEIGHT = 32

# Compute focal length
focal = 0.5 * CIFAR_WIDTH / math.tan(0.5 * math.pi / 2.0)

# Camera intrinsic matrix for CIFAR-10
CIFAR10_KS = torch.tensor(
    [
        [focal, 0, CIFAR_WIDTH / 2],
        [0, focal, CIFAR_HEIGHT / 2],
        [0, 0, 1],
    ],
    dtype=torch.float32,  # Ensure the tensor is float32
)

# View matrix for CIFAR-10
CIFAR10_VIEWMATS = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 8.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,  # Ensure the tensor is float32
)

CIFAR10_GRID_RANGES = {
    "means": {
        "min": -1.0,
        "max": 1.0,
        "mean": torch.tensor([3.0734e-08, 0.0000e00, 0.0000e00]),
        "std": torch.tensor([0.5957, 0.5957, 0.0000]),
    },
    "quats": {
        "min": -3.7537035942077637,
        "max": 4.574342727661133,
        "mean": torch.tensor([0.4909, 0.5113, 0.4963, 0.4855]),
        "std": torch.tensor([0.4740, 0.5310, 0.5126, 0.4743]),
    },
    "scales": {
        "min": -14.256706237792969,
        "max": 6.657063961029053,
        "mean": torch.tensor([-1.8530, -1.8229, -2.4988]),
        "std": torch.tensor([1.4585, 1.5092, 0.5869]),
    },
    "opacities": {
        "min": -5.512201309204102,
        "max": 7.002721309661865,
        "mean": torch.tensor(-3.3678),
        "std": torch.tensor(1.4174),
    },
    "colors": {
        "min": -15.537788391113281,
        "max": 17.288856506347656,
        "mean": torch.tensor(
            [
                [0.6861, 0.6412, 0.6534],
                [-0.0346, -0.0275, -0.0154],
                [0.7585, 0.7298, 0.6635],
                [-0.0100, -0.0208, -0.0225],
            ]
        ),
        "std": torch.tensor(
            [
                [2.4589, 2.3819, 2.4028],
                [1.9165, 1.8434, 1.8344],
                [2.1524, 2.0766, 2.0886],
                [1.9149, 1.8357, 1.8274],
            ]
        ),
    },
}
