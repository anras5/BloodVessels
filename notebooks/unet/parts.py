import torch
import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    """Convolutional block for UNet.
    Consists of two Convolutional layers with 3x3 kernel and 1 pixel padding.
    Each Convolutional layer is ended with ReLU activation function.

    Useful links:
    1. https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),  # original U-Net does not use batch normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Down Block consists of `ConvolutionalBlock` + MaxPooling layer with 2x2 kernel"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvolutionalBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x, self.pool(x)


class UpBlock(nn.Module):
    """Up Block consists of `ConvTranspose` layer with 2x2 kernel and 2 pixels stride"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvolutionalBlock(in_c, out_c)

    def forward(self, inputs, skipped):
        x = self.up(inputs)
        x = torch.cat([x, skipped], dim=1)
        return self.conv(x)


class OutBlock(nn.Module):
    """Out Block is the final layer of the UNet.
    It consists of one Convolutional Layer with 1x1 kernel and sigmoid activation function.
    """
    def __init__(self, in_c, out_c):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        return self.sig(x)
