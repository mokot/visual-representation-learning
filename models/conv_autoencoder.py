import torch.nn as nn
from .abstract_autoencoder import AbstractAutoencoder


class ConvAutoencoder(AbstractAutoencoder):

    def __init__(
        self,
        channels_dim: int,
        use_bias: bool = True,
        weight_init: bool = True,
    ):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels_dim, 32, kernel_size=3, padding=1, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, output_padding=1, bias=use_bias
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, output_padding=1, bias=use_bias
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                32,
                channels_dim,
                kernel_size=3,
                stride=2,
                output_padding=1,
                bias=use_bias,
            ),
            nn.Tanh(),  # Output [-1, 1]
        )
        if weight_init:
            self.init_weights()
