#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from SegNet import *
from DeeplabLargeFOV import *


factory = {
        'SegNet': SegNet,
        'DeeplabLargeFOV': DeepLabLargeFOV,
        }


def get_model(cfg):
    model = cfg.model
    if model.backbone in factory.keys():
        return factory[model.backbone](in_dim = model.in_channel,
                out_dim = model.num_class)
    else:
        raise NameError("unsupported model type {}".format(model_name))


if __name__ == "__main__":
    import torch
    segnet = get_model("SegNet").float().cuda()
    in_ten = torch.randn((1, 3, 224, 224)).cuda()
    out = segnet(in_ten)
    print(out.detach().cpu().numpy())

    deeplablargefov = get_model("DeeplabLargeFOV", 3, 21).float().cuda()
    in_ten = torch.randn((1, 3, 224, 224)).cuda()
    out = deeplablargefov(in_ten)
    print(out.detach().cpu().numpy())
