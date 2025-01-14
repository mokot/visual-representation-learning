import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Optional, List, Dict


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: Callable,
    epochs: int,
    device: torch.device,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    grad_clip: Optional[float] = None,
    logger: Optional[Callable[[str], None]] = print,
) -> Dict[str, List[float]]:
    """
    Train the model on the training dataset and evaluate it on the validation dataset.

    Args:
        model (nn.Module): The autoencoder model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (Optimizer): Optimizer for updating model weights.
        criterion (Callable): Loss function.
        epochs (int): Number of training epochs.
        device (torch.device): Device to use for computation (e.g., 'cuda' or 'cpu').
        scheduler (Optional[LRScheduler]): Learning rate scheduler.
        grad_clip (Optional[float]): Gradient clipping threshold (if any).
        logger (Optional[Callable[[str], None]]): Function for logging progress (e.g., `print` or a custom logger).

    Returns:
        Dict[str, List[float]]: Dictionary containing training and validation losses for each epoch.
    """
    model.to(device)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                x_hat = model(x)
                loss = criterion(x_hat, x)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

        if scheduler:
            scheduler.step(val_loss)

        # Logging
        log_message = f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        if logger:
            logger(log_message)

    return history


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: Callable,
    device: torch.device,
    logger: Optional[Callable[[str], None]] = print,
) -> float:
    """
    Evaluate the model on a dataset.

    Args:
        model (nn.Module): The autoencoder model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation set.
        criterion (Callable): Loss function.
        device (torch.device): Device to use for computation (e.g., 'cuda' or 'cpu').
        logger (Optional[Callable[[str], None]]): Function for logging progress.

    Returns:
        float: Average loss on the dataset.
    """
    model.eval()
    loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            x_hat = model(x)
            loss += criterion(x_hat, x).item()

    avg_loss = loss / len(data_loader)
    if logger:
        logger(f"Evaluation Loss: {avg_loss:.4f}")
    return avg_loss
