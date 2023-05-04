import torch
import torch.nn as nn

from notebooks.unet.parts import DownBlock, ConvolutionalBlock, UpBlock, OutBlock


class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()

        self.down1 = DownBlock(n_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        self.b = ConvolutionalBlock(512, 1024)

        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        self.c = OutBlock(64, 1)

    def forward(self, inputs):
        l1, p1 = self.down1(inputs)
        l2, p2 = self.down2(p1)
        l3, p3 = self.down3(p2)
        l4, p4 = self.down4(p3)

        b = self.b(p4)

        d1 = self.up1(b, l4)
        d2 = self.up2(d1, l3)
        d3 = self.up3(d2, l2)
        d4 = self.up4(d3, l1)

        output = self.c(d4)

        return output
