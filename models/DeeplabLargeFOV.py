#!/usr/bin/python
# -*- encoding: utf-8 -*-


from common import ConvReLU
import torchvision
import torch
import torch.nn as nn


## TODO:
# 1. minus sample means
# 2. different lrs for different parameters

'''
Notes:
    1. an avgpool with stride = 1 is inserted closely after the last maxpool
    2. the original fc layers are replaced with conv layers in which the first
    fc layer has large dilation (12) and the others have kernel size of 1.
    For the large dilation fc layer, the padding is also enlarged to 12x(3-1)/2=12 for sake of holding still the feature
    map sizes. Provided the kernel size is 4, then the padding should be computed
    to be 12x(4-1)/2=18
'''



class DeepLabLargeFOV(nn.Module):
    def __init__(self, in_channel, out_num, *args, **kwargs):
        super(DeepLabLargeFOV, self).__init__(*args, *kwargs)
        self.conv1_1 = ConvReLU(in_channel, 64, kernel = 3, stride = 1, pad = 1)
        self.conv1_2 = ConvReLU(64, 64, kernel = 3, stride = 1, pad = 1)
        self.pool1 = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.conv2_1 = ConvReLU(64, 128, kernel = 3, stride = 1, pad = 1)
        self.conv2_2 = ConvReLU(128, 128, kernel = 3, stride = 1, pad = 1)
        self.pool2 = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.conv3_1 = ConvReLU(128, 256, kernel = 3, stride = 1, pad = 1)
        self.conv3_2 = ConvReLU(256, 256, kernel = 3, stride = 1, pad = 1)
        self.conv3_3 = ConvReLU(256, 256, kernel = 3, stride = 1, pad = 1)
        self.pool3 = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.conv4_1 = ConvReLU(256, 512, kernel = 3, stride = 1, pad = 1)
        self.conv4_2 = ConvReLU(512, 512, kernel = 3, stride = 1, pad = 1)
        self.conv4_3 = ConvReLU(512, 512, kernel = 3, stride = 1, pad = 1)
        self.pool4 = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.conv5_1 = ConvReLU(512, 512, kernel = 3, stride = 1, pad = 2, dilation = 2)
        self.conv5_2 = ConvReLU(512, 512, kernel = 3, stride = 1, pad = 2, dilation = 2)
        self.conv5_3 = ConvReLU(512, 512, kernel = 3, stride = 1, pad = 2, dilation = 2)
        self.pool5 = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.pool5a = nn.AvgPool2d(3, stride = 1, padding = 1)
        self.fc6 = ConvReLU(512, 1024, kernel = 3, stride = 1, pad = 12, dilation = 12)
        self.drop6 = nn.Dropout(p = 0.5)
        self.fc7 = ConvReLU(1024, 1024, kernel = 1, stride = 1, pad = 0)
        self.drop7 = nn.Dropout(p = 0.5)
        self.fc8 = nn.Conv2d(1024, out_num, kernel_size = 1)

        self.init_params()


    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        x = self.pool5a(x)
        x = self.fc6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.drop7(x)
        x = self.fc8(x)
        return x


    def init_params(self):
        vgg16_pretrained = torchvision.models.vgg16(pretrained = True)
        new_dict = dict()
        key_iter = iter(list(self.state_dict().keys()))
        for layer in vgg16_pretrained.features:
            if isinstance(layer, nn.Conv2d):
                for i in range(2):
                    k = next(key_iter)
                    v = list(layer.parameters())[i]
                    new_dict.update({k: v})
        self.state_dict().update(new_dict)

        nn.init.normal_(self.fc8.weight, 0, 0.01)
        nn.init.constant_(self.fc8.bias, 0)


if __name__ == '__main__':
    model = DeepLabLargeFOV(3, 10).float().cuda()
    import numpy as np
    in_data = np.random.randn(10, 3, 500, 600)
    in_tensor = torch.tensor(in_data, dtype = torch.float32).cuda()
    out = model(in_tensor).detach().cpu().numpy()
    print(out)
    print(out.shape)
    #
    #  import torchvision
    #  vgg16 = torchvision.models.vgg16(pretrained = True)
    #  count = 0
    #  for name, params in vgg16.features.named_parameters():
    #      print(name)
    #      count += 1
    #  print(count)
    #  count = 0
    #  for name, params in model.named_parameters():
    #      print(name)
    #      if 'conv' in name:
    #          count += 1
    #  print(count)
    #  print(vgg16)
    #  print(model.state_dict())


