import torch.nn as nn


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """Creates a 3x3 convolutional layer with optional stride, groups, and dilation.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        groups (int, optional): Number of blocked connections. Defaults to 1.
        dilation (int, optional): Dilation factor for spacing between kernel elements. Defaults to 1.

    Returns:
        nn.Conv2d: A 3x3 convolutional layer with padding and no bias.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,  # Ensures output size consistency
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Creates a 1x1 convolutional layer, often used for dimensionality reduction or feature transformation.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to 1.

    Returns:
        nn.Conv2d: A 1x1 convolutional layer without bias.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


def conv3x3Transposed(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    output_padding: int = 0,
) -> nn.ConvTranspose2d:
    """Creates a 3x3 transposed convolutional (deconvolution) layer with padding.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        groups (int, optional): Number of blocked connections. Defaults to 1.
        dilation (int, optional): Dilation factor for spacing between kernel elements. Defaults to 1.
        output_padding (int, optional): Extra size added to the output tensor. Required for inverting strided convs. Defaults to 0.

    Returns:
        nn.ConvTranspose2d: A 3x3 transposed convolutional layer.
    """
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        output_padding=output_padding,  # Necessary for inverting conv2d with stride > 1
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1Transposed(
    in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0
) -> nn.ConvTranspose2d:
    """Creates a 1x1 transposed convolutional (deconvolution) layer.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        output_padding (int, optional): Extra size added to the output tensor. Required for inverting strided convs. Defaults to 0.

    Returns:
        nn.ConvTranspose2d: A 1x1 transposed convolutional layer.
    """
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        output_padding=output_padding,
    )
