import torch
from sklearn.neighbors import NearestNeighbors


def compute_knn_distances(x: torch.Tensor, k: int = 4) -> torch.Tensor:
    """
    Computes k-nearest neighbors distances for the given tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (N, D), where N is the number of points and D is the dimension.
        k (int): Number of neighbors to consider (default: 4).

    Returns:
        torch.Tensor: Tensor of distances of shape (N, k).
    """
    # Convert tensor to NumPy array
    x_np = x.cpu().numpy()

    # Fit the NearestNeighbors model
    model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    model.fit(x_np)

    # Compute k-nearest neighbors
    distances, _ = model.kneighbors(x_np)

    # Convert distances back to a PyTorch tensor
    return torch.from_numpy(distances).to(x.device)
