#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from config import load_cfg
from model import SegNet
import argparse
import cv2
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wts', dest='wts', required=True,
        help='model weights to be loaded')
    parser.add_argument(
        '--cfg', dest='cfg', required=True,
        help='configure file to be loaded')
    parser.add_argument(
        'im_path', help='image path', default=None
    )
    return parser.parse_args()



def test(args):
    cfg = load_cfg(args.cfg)
    weight_path = args.wts
    img_path = args.im_path

    segnet = SegNet().float().cuda()
    segnet.load_state_dict(torch.load(weight_path))
    segnet.eval()

    im = cv2.imread(img_path).transpose(2, 0, 1)
    im = torch.tensor(im[np.newaxis, :], dtype=torch.float).cuda()
    out = segnet(im)
    out = out.detach().cpu().numpy().transpose(0, 2, 3, 1)
    out = np.argmax(out, axis=3).astype(np.uint8)[0]
    out = out[:, :, np.newaxis]
    out = out * 20
    cv2.imshow('fuck', out)
    cv2.waitKey(0)



if __name__ == "__main__":
    args = get_args()
    test(args)
