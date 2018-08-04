#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn

# TODO:
# 0. 各层尤其是conv层怎么初始化的。
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

        self.classifier = nn.Conv2d(64, 12, 1, 1)


    def forward(self, x):
        x = self.conv1(x)
        size1 = x.size()
        x, idx1 = self.pool1(x)
        x = self.conv2(x)
        size2 = x.size()
        x, idx2 = self.pool2(x)
        x = self.conv3(x)
        size3 = x.size()
        x, idx3 = self.pool3(x)
        x = self.conv4(x)
        size4 = x.size()
        x, idx4 = self.pool4(x)

        x = self.upsample4(x, idx4, output_size = size4)
        x = self.deconv4(x)
        x = self.upsample3(x, idx3, output_size = size3)
        x = self.deconv3(x)
        x = self.upsample2(x, idx2, output_size = size2)
        x = self.deconv2(x)
        x = self.upsample1(x, idx1, output_size = size1)
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


if __name__ == "__main__":
    segnet = SegNet().cuda()
    segnet.float()
    import numpy as np
    in_array = np.random.randn(4, 3, 360, 480)
    in_tensor = torch.as_tensor(in_array, dtype=torch.float32).cuda()
    in_tensor.float()
    print(type(in_tensor))
    print(in_tensor.dtype)
    segnet.train()
    out_tensor = segnet(in_tensor)
    print(out_tensor.data)

