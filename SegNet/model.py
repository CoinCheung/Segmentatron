#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn

# TODO:
# 1. BN 是怎么初始化的
# 2. 数据是否需要预处理
# 3. pool是same 还是valid --- 就是 2, 2
# 4. 训练时不同层的lr_mut不同
# 5. upsample时候的scale是2，要加吗
# 6. 是否使用了data aug

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.conv1 = self.ConvBNReLU(3, 64, 7, 1, 3)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv2 = self.ConvBNReLU(64, 64, 7, 1, 3)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv3 = self.ConvBNReLU(64, 64, 7, 1, 3)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv4 = self.ConvBNReLU(64, 64, 7, 1, 3)
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        self.upsample4 = nn.MaxUnpool2d(2, 2)
        self.deconv4 = self.ConvBN(64, 64, 7, 1, 3)

        self.upsample3 = nn.MaxUnpool2d(2, 2)
        self.deconv3 = self.ConvBN(64, 64, 7, 1, 3)

        self.upsample2 = nn.MaxUnpool2d(2, 2)
        self.deconv2 = self.ConvBN(64, 64, 7, 1, 3)

        self.upsample1 = nn.MaxUnpool2d(2, 2)
        self.deconv1 = self.ConvBN(64, 64, 7, 1, 3)

        self.classifier = nn.Conv2d(64, 11, 1, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.upsample4(x)
        x = self.deconv4(x)
        x = self.upsample3(x)
        x = self.deconv3(x)
        x = self.upsample2(x)
        x = self.deconv2(x)
        x = self.upsample1(x)
        x = self.deconv1(x)
        x = self.classifier(x)
        return x


    def ConvBNReLU(self, in_channel, out_channel, kernel, stride, pad):
        return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel, stride, pad),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
                )


    def ConvBN(self, in_channel, out_channel, kernel, stride, pad):
        return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel, stride, pad),
                nn.BatchNorm2d(out_channel)
                )


