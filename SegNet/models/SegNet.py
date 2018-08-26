#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn

from common import ConvBNReLU, ConvBN

# TODO:
# 4. 训练时不同层的lr_mut不同

class SegNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SegNet, self).__init__(*args, **kwargs)
        self.norm = nn.LocalResponseNorm(5, alpha = 0.0001, beta = 0.75)
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
        ## It turns out that the model behaves better without this LRN layer
        #  x = self.norm(x)
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


if __name__ == "__main__":
    segnet = SegNet().cuda()
    segnet.float()
    import numpy as np
    in_array = np.random.randn(4, 3, 360, 480)
    in_tensor = torch.tensor(in_array, dtype=torch.float32).cuda()
    in_tensor.float()
    print(type(in_tensor))
    print(in_tensor.dtype)
    segnet.train()
    out_tensor = segnet(in_tensor)
    print(out_tensor.data)
