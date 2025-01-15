import torch.nn as nn
from .abstract_autoencoder import AbstractAutoencoder


class ShallowAutoencoder(AbstractAutoencoder):
    def __init__(self, input_dim: int, latent_dim: int, use_bias: bool = True):
        super(ShallowAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim, bias=use_bias),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim, bias=use_bias),
            nn.Sigmoid(),
        )
