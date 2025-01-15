import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Optional, List, Dict, Union, Tuple


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epochs: int,
    device: torch.device,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    grad_clip: Optional[float] = None,
    logger: Optional[Callable[[str], None]] = print,
    compile_model: bool = False,
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
        compile_model (bool): If True, compiles the model with `torch.compile` for optimization.

    Returns:
        Dict[str, List[float]]: Dictionary containing training and validation losses for each epoch.
    """
    if compile_model:
        model = torch.compile(model)

    model.to(device)
    history = {"train_loss": [], "val_loss": []}

    # Patience for early stopping
    patience = 5
    patience_counter = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Shuffle the training data
        train_loader.dataset.shuffle()

        # Training phase
        model.train()
        train_loss = 0.0
        for x in tqdm.tqdm(
            train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            # TODO: check that x has not changed!
            loss = criterion(x_hat, x)
            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation phase
        val_loss = evaluate(model, val_loader, criterion, device, None)
        history["val_loss"].append(val_loss)

        # Scheduler step
        if scheduler:
            scheduler.step(val_loss)

        # Logging
        if logger:
            log_message = f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            logger(log_message)

        # Early stopping (patience)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == patience:
                if logger:
                    logger(f"Stopping early after {epoch + 1} epochs.")
                # TODO: save the best model
                break

    # TODO: save the best model
    return history


def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: Callable,
    device: torch.device,
    logger: Optional[Callable[[str], None]] = print,
    compile_model: bool = False,
) -> float:
    """
    Evaluate the model on a dataset.

    Args:
        model (nn.Module): The autoencoder model to evaluate.
        val_loader (DataLoader): DataLoader for the evaluation set.
        criterion (Callable): Loss function.
        device (torch.device): Device to use for computation (e.g., 'cuda' or 'cpu').
        logger (Optional[Callable[[str], None]]): Function for logging progress.
        compile_model (bool): If True, compiles the model with `torch.compile` for optimization.

    Returns:
        float: Average loss on the dataset.
    """
    if compile_model:
        model = torch.compile(model)

    model.to(device)
    model.eval()
    loss = 0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            x_hat = model(x)
            loss += criterion(x_hat, x).item()

    avg_loss = loss / len(val_loader)
    if logger:
        logger(f"Evaluation Loss: {avg_loss:.4f}")
    return avg_loss


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: Callable,
    device: torch.device,
    return_samples: bool = False,
    logger: Optional[Callable[[str], None]] = print,
    compile_model: bool = False,
) -> Union[float, Tuple[float, Tuple[List[torch.Tensor], List[torch.Tensor]]]]:
    """
    Test the model on the test dataset.

    Args:
        model (nn.Module): The autoencoder model to test.
        test_loader (DataLoader): DataLoader for the test set.
        criterion (Callable): Loss function.
        device (torch.device): Device to use for computation (e.g., 'cuda' or 'cpu').
        return_samples (bool): If True, return the original and reconstructed samples for further analysis.
        logger (Optional[Callable[[str], None]]): Function for logging progress (e.g., `print` or a custom logger).
        compile_model (bool): If True, compiles the model with `torch.compile` for optimization.

    Returns:
        float: Average loss on the test dataset.
        Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]: If `return_samples` is True, returns a tuple of original and reconstructed samples.
    """
    if compile_model:
        model = torch.compile(model)

    model.to(device)
    model.eval()
    loss = 0.0
    original_samples = []
    reconstructed_samples = []

    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            x_hat = model(x)
            batch_loss = criterion(x_hat, x)
            loss += batch_loss.item()

            if return_samples:
                original_samples.append(x.cpu())
                reconstructed_samples.append(x_hat.cpu())

    avg_loss = loss / len(test_loader)

    if logger:
        logger(f"Test Loss: {avg_loss:.4f}")

    if return_samples:
        return avg_loss, (original_samples, reconstructed_samples)

    return avg_loss
