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
    "means": {"min": -1.0, "max": 1.0},
    "quats": {"min": -3.7537035942077637, "max": 4.574342727661133},
    "scales": {"min": -14.256706237792969, "max": 6.657063961029053},
    "opacities": {"min": -5.512201309204102, "max": 7.002721309661865},
    "colors": {"min": -15.537788391113281, "max": 17.288856506347656},
}
