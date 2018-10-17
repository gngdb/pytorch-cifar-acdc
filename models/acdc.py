# a tiny CNN for testing 
# simple training got 94% train and 89% validation on CIFAR-10, validation loss 0.33

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tiny import AllConv
from .resnet import ResNet, Bottleneck
from .wide_resnet import BasicBlock, WideResNet
from torch_dct.layers import FastStackedConvACDC

def Conv2d(in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
    return FastStackedConvACDC(in_channels, out_channels, kernel_size, 12,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)

def AllConvACDC():
    return AllConv(Conv2d=Conv2d)

def ResNetACDC(plane_multiplier):
    return ResNet(Bottleneck, [3,4,6,3], Conv2d=Conv2d, plane_multiplier=plane_multiplier)

def WideResNetACDC(depth, width):
    return WideResNet(depth,width,Conv2d,BasicBlock)
