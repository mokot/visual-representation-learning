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
