import torch
import torch.nn as nn
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, Optional, List, Dict
from utils import (
    train as train_autoencoder,
    evaluate as evaluate_autoencoder,
    test as test_autoencoder,
)


class AbstractAutoencoder(nn.Module, ABC):

    @abstractmethod
    def __init__(self: nn.Module) -> None:
        """
        Initializes the base autoencoder class.
        """
        super(AbstractAutoencoder, self).__init__()
        self.encoder = None
        self.decoder = None

    def forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the autoencoder: encode -> decode.
        """
        return self.decode(self.encode(x))

    def encode(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input using the encoder.
        """
        if self.encoder is None:
            raise NotImplementedError("Encoder is not defined.")
        return self.encoder(x)

    def decode(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input using the decoder.
        """
        if self.decoder is None:
            raise NotImplementedError("Decoder is not defined.")
        return self.decoder(x)

    def fit(
        self: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        epochs: int,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: Optional[float] = None,
        logger: Optional[Callable[[str], None]] = print,
    ) -> Dict[str, List[float]]:
        """
        Trains the autoencoder using the utils fit function.
        Note: Cannot use `train` due to conflict with `nn.Module.train()`.
        """
        return train_autoencoder(
            model=self,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            device=device,
            scheduler=scheduler,
            grad_clip=grad_clip,
            logger=logger,
        )

    def evaluate(
        self: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: Callable,
        device: torch.device,
        logger: Optional[Callable[[str], None]] = print,
    ) -> float:
        """
        Evaluates the autoencoder using the utils evaluate function.
        """
        return evaluate_autoencoder(
            model=self,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            logger=logger,
        )

    def test(
        self: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        criterion: Callable,
        device: torch.device,
        return_samples: bool = False,
        logger: Optional[Callable[[str], None]] = print,
    ) -> float:
        """
        Tests the autoencoder using the utils test function.
        """
        return test_autoencoder(
            model=self,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            return_samples=return_samples,
            logger=logger,
        )

    def save(self: nn.Module, path: Path) -> None:
        """
        Saves the model state dictionary to the specified path.
        """
        torch.save(self.state_dict(), path)

    def load(self: nn.Module, path: Path) -> None:
        """
        Loads the model state dictionary from the specified path.
        """
        self.load_state_dict(torch.load(path))

    def save_model(self: nn.Module, path: Path) -> None:
        """
        Saves the entire model to the specified path.
        """
        torch.save(self, path)

    @staticmethod
    def load_model(path: Path) -> nn.Module:
        """
        Loads an entire model from the specified path.
        """
        return torch.load(path)
