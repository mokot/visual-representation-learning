import torch.nn as nn
from .abstract_autoencoder import AbstractAutoencoder


class DeepAutoencoder(AbstractAutoencoder):

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        use_bias: bool = True,
        weight_init: bool = True,
    ):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8192, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 4096, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(2048, latent_dim, bias=use_bias),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4096, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 8192, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(8192, input_dim, bias=use_bias),
            nn.Tanh(),  # Values are in the range [-1, 1]
        )
        if weight_init:
            self.init_weights()
