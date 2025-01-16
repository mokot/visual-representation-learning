import torch
import torch.nn as nn
from pathlib import Path
from abc import ABC, abstractmethod


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

    def init_weights(self):
        """
        Initializes the weights of the model using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
