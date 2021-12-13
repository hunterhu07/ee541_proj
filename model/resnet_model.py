#!/usr/bin/env python
# coding=utf-8
# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from torchvision.models import resnet18
from torch import nn

import torch


class ResNet(nn.Module):
    def __init__(self, n_classes=2):
        super(ResNet, self).__init__()
        self.resnet = resnet18()  # (pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
