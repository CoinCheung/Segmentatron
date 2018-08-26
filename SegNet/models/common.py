#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn



class ConvBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad, dilation = 1,
            *args, **kwargs):
        super(ConvBNReLU, self).__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, pad, dilation)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad, dilation = 1,
            *args, **kwargs):
        super(ConvBN, self).__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, pad, dilation)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad, dilation = 1,
            *args, **kwargs):
        super(ConvReLU, self).__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, pad, dilation)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
