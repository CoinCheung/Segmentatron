#!/usr/bin/python
# -*- encoding: utf-8 -*-


from datasets import get_datasets
from model import SegNet
from torch.utils.data import DataLoader
import torch.nn as nn

import torch
import json





with open('./config.json', 'r') as jr:
    cfg = json.load(jr)

print(cfg)


def train():
    trainset = get_datasets(cfg['datasets'], 'train')
    valset = get_datasets(cfg['datasets'], 'val')
    trainloader = DataLoader(trainset,
                            batch_size = 4,
                            shuffle = True,
                            num_workers = 4)
    valloader = DataLoader(valset,
                            batch_size = 4,
                            shuffle = True,
                            num_workers = 4)

    segnet = SegNet()
    Loss = nn.CrossEntropyLoss()

    for im, label in trainloader:
        print(im.shape)
        logits = segnet(im)
        loss = Loss(logits, label)

        print(loss.detach().numpy())

        break

    #  logits = segnet(x)

    #  import cv2
    #  for im, label in train_loader:
    #      cv2.imshow('im', im[0].numpy())
    #      cv2.imshow('label', 50 * label[0].numpy())
    #      cv2.waitKey(0)


if __name__ == '__main__':
    train()
