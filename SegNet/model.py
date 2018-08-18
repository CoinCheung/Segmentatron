#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn

# TODO:
# 0. 各层尤其是conv层怎么初始化的。 -- conv初始化了，BN 好像没法初始化
# 1. BN 是怎么初始化的
# 2. 数据是否需要预处理 -- 没有，只有一个LRN，但是没什么用
# 3. pool是same 还是valid --- 就是 2, 2
# 4. 训练时不同层的lr_mut不同
# 5. upsample时候的scale是2，要加吗 -- done，maxunpool已经指明了
# 6. 是否使用了data aug -- 没有，有LRN，但是没用。

class SegNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SegNet, self).__init__(*args, **kwargs)
        self.conv1 = ConvBNReLU(3, 64, 7, 1, 3)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv2 = ConvBNReLU(64, 64, 7, 1, 3)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv3 = ConvBNReLU(64, 64, 7, 1, 3)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv4 = ConvBNReLU(64, 64, 7, 1, 3)
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        self.upsample4 = nn.MaxUnpool2d(2, 2)
        self.deconv4 = ConvBN(64, 64, 7, 1, 3)

        self.upsample3 = nn.MaxUnpool2d(2, 2)
        self.deconv3 = ConvBN(64, 64, 7, 1, 3)

        self.upsample2 = nn.MaxUnpool2d(2, 2)
        self.deconv2 = ConvBN(64, 64, 7, 1, 3)

        self.upsample1 = nn.MaxUnpool2d(2, 2)
        self.deconv1 = ConvBN(64, 64, 7, 1, 3)

        self.classifier = nn.Conv2d(64, 12, 1, 1)

        self.init_params()


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

    def init_params(self):
        for name, param in self.named_parameters():
            if 'conv' in name.split('.')[1:]:
                if 'weight' in name:
                    nn.init.kaiming_normal_(param, 'fan_out', nonlinearity = 'relu')
                elif 'bias' in name:
                    nn.init.constant_(param, 0)



class ConvBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad,
            *args, **kwargs):
        super(ConvBNReLU, self).__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, pad)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad,
            *args, **kwargs):
        super(ConvBN, self).__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, pad)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


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

