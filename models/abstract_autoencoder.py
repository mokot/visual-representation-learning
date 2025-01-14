import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Callable, Optional, List, Dict
from utils import train as train_autoencoder, evaluate as evaluate_autoencoder


class AbstractAutoencoder(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        """
        Initializes the base autoencoder class.
        """
        super(AbstractAutoencoder, self).__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, x):
        """
        Passes the input through the autoencoder: encode -> decode.
        """
        return self.decode(self.encode(x))

    def encode(self, x):
        """
        Encodes the input using the encoder.
        """
        if self.encoder is None:
            raise NotImplementedError("Encoder is not defined.")
        return self.encoder(x)

    def decode(self, x):
        """
        Decodes the input using the decoder.
        """
        if self.decoder is None:
            raise NotImplementedError("Decoder is not defined.")
        return self.decoder(x)

    def fit(
        self,
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
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: Callable,
        device: torch.device,
        logger: Optional[Callable[[str], None]] = print,
    ) -> float:
        """
        Evaluates the autoencoder using the utils evaluate function.
        """
        return evaluate_autoencoder(
            model=self,
            data_loader=data_loader,
            criterion=criterion,
            device=device,
            logger=logger,
        )

    @abstractmethod
    def loss_function(self, x, x_hat):
        """
        Computes the loss given the original input and its reconstruction.
        """
        pass

    @abstractmethod
    def sample(self, num_samples):
        """
        Generates samples from the latent space.
        """
        pass

    @abstractmethod
    def generate(self, x):
        """
        Generates new data based on input.
        """
        pass

    @abstractmethod
    def reconstruct(self, x):
        """
        Reconstructs the input using the autoencoder.
        """
        pass

    @abstractmethod
    def get_latent(self, x):
        """
        Extracts the latent representation of the input.
        """
        pass

    @abstractmethod
    def get_latent_dim(self):
        """
        Returns the dimensionality of the latent space.
        """
        pass

    def get_name(self):
        """
        Returns the name of the model.
        """
        return self.__class__.__name__

    def get_num_params(self):
        """
        Returns the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_device(self):
        """
        Returns the device on which the model is located.
        """
        if not list(self.parameters()):
            raise ValueError("Model has no parameters.")
        return next(self.parameters()).device

    def to_device(self, device):
        """
        Moves the model to the specified device.
        """
        self.to(device)

    def save(self, path):
        """
        Saves the model state dictionary to the specified path.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Loads the model state dictionary from the specified path.
        """
        self.load_state_dict(torch.load(path))

    def save_model(self, path):
        """
        Saves the entire model to the specified path.
        """
        torch.save(self, path)

    @staticmethod
    def load_model(path):
        """
        Loads an entire model from the specified path.
        """
        return torch.load(path)

    def __str__(self):
        """
        Returns a string representation of the model.
        """
        try:
            return f"{self.get_name()} ({self.get_num_params()} params)"
        except NotImplementedError:
            return f"AbstractAutoencoder ({self.get_num_params()} params)"

    def __repr__(self):
        return self.__str__()
