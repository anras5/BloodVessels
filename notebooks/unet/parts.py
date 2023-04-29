import torch
import torch.nn as nn


class ConvolutionalBlock(nn.Module):
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
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvolutionalBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x, self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvolutionalBlock(in_c, out_c)

    def forward(self, inputs, skipped):
        x = self.up(inputs)
        x = torch.cat([x, skipped], dim=1)
        return self.conv(x)


class OutBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
