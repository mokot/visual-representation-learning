import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Type
from utils.conv import conv3x3, conv3x3Transposed
from .abstract_autoencoder import AbstractAutoencoder


class ResNetAutoencoder(AbstractAutoencoder):

    def __init__(
        self,
        channels_dim: int,
        weight_init: bool = True,
    ):
        super(ResNetAutoencoder, self).__init__()
        self.encoder = ResNetEncoder(channels_dim, [2, 2, 2], weight_init)
        self.decoder = ResNetDecoder(channels_dim, [2, 2, 2], weight_init)


class ResNetEncoderBlock(nn.Module):
    """Basic ResNet encoder block with two convolutional layers."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):
    """ResNet-based encoder with multiple convolutional layers."""

    def __init__(
        self,
        in_channels: int,
        layers: List[int],
        zero_init_residual: bool,
    ) -> None:
        super().__init__()
        self.in_channels = 16  # Initial number of channels

        # Initial convolutional layer
        self.conv1 = conv3x3(in_channels, self.in_channels)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Create ResNet blocks
        self.layer1 = self._make_layer(ResNetEncoderBlock, 16, layers[0])
        self.layer2 = self._make_layer(ResNetEncoderBlock, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(ResNetEncoderBlock, 64, layers[2], stride=2)

        # Initialize weights
        self._initialize_weights(zero_init_residual)

    def _make_layer(
        self,
        block: Type[ResNetEncoderBlock],
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Creates a ResNet layer with multiple blocks."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual: bool) -> None:
        """Initializes model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetEncoderBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the encoder."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class ResNetDecoderBlock(nn.Module):
    """Basic ResNet decoder block with two transposed convolutional layers."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        output_padding: int = 0,
        upsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3Transposed(
            out_channels, in_channels, stride, output_padding=output_padding
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3Transposed(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = upsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.upsample is not None:
            identity = self.upsample(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNetDecoder(nn.Module):
    """ResNet-based decoder with multiple transposed convolutional layers."""

    def __init__(
        self,
        out_channels: int,
        layers: List[int],
        zero_init_residual: bool,
    ) -> None:
        super().__init__()
        self.in_channels = 16  # Start with the highest feature map size

        # Create ResNet blocks (decoder layers)
        self.layer3 = self._make_layer(ResNetDecoderBlock, 64, layers[2], stride=2)
        self.layer2 = self._make_layer(ResNetDecoderBlock, 32, layers[1], stride=2)
        self.layer1 = self._make_layer(
            ResNetDecoderBlock,
            16,
            layers[0],
            stride=1,
            output_padding=0,
            last_block_dim=16,
        )

        # Final transposed convolution to reconstruct output
        self.deconv1 = conv3x3Transposed(self.in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Initialize weights
        self._initialize_weights(zero_init_residual)

    def _make_layer(
        self,
        block: Type[ResNetDecoderBlock],
        out_channels: int,
        num_blocks: int,
        stride: int = 2,
        output_padding: int = 1,
        last_block_dim: int = 0,
    ) -> nn.Sequential:
        """Creates a ResNet decoder layer with multiple blocks."""
        layers = []
        upsample = None
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        if last_block_dim == 0:
            last_block_dim = self.in_channels // 2

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            upsample = nn.Sequential(
                conv3x3Transposed(
                    self.in_channels,
                    last_block_dim,
                    stride,
                    output_padding=1,
                ),
                nn.BatchNorm2d(last_block_dim),
            )

        layers.append(
            block(last_block_dim, out_channels, stride, output_padding, upsample)
        )

        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual: bool) -> None:
        """Initializes model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetDecoderBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the decoder."""
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
