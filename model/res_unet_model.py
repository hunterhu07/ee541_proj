#!/usr/bin/env python
# coding=utf-8
# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *
from .resnet_model import ResNet


class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResUNet, self).__init__()
        self.resnet = ResNet(n_classes=2)
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.up1_0 = up(1024, 256)
        self.up2_0 = up(512, 128)
        self.up3_0 = up(256, 64)
        self.up4_0 = up(128, 64)
        self.outc_0 = outconv(64, n_classes)

        self.up1_1 = up(1024, 256)
        self.up2_1 = up(512, 128)
        self.up3_1 = up(256, 64)
        self.up4_1 = up(128, 64)
        self.outc_1 = outconv(64, n_classes)

    def forward(self, x):
        x_res = self.resnet(x)
        cell_type = torch.argmax(x_res, dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_0 = self.up1_0(x5, x4)
        x_0 = self.up2_0(x_0, x3)
        x_0 = self.up3_0(x_0, x2)
        x_0 = self.up4_0(x_0, x1)
        x_0 = self.outc_0(x_0)
        x_0 = F.sigmoid(x_0)

        x_1 = self.up1_1(x5, x4)
        x_1 = self.up2_1(x_1, x3)
        x_1 = self.up3_1(x_1, x2)
        x_1 = self.up4_1(x_1, x1)
        x_1 = self.outc_1(x_1)
        x_1 = F.sigmoid(x_1)

        x_l = []
        for i, c in enumerate(cell_type):
            if c == 0:
                x_l.append(x_0[i])
            else:
                x_l.append(x_1[i])
        x_c = torch.stack(x_l, dim=0)

        return x_c, x_res
