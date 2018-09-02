#!/usr/bin/python
# -*- encoding: utf-8 -*-


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import CamVid
import PascalVoc


dataset_names = ['CamVid', 'PascalVoc2012']


def get_datasets(dname, mode):
    assert dname in dataset_names, 'unknown dataset: {}'.format(dname)
    if dname == 'CamVid':
        if mode == 'train' or mode == 'val' or mode == 'test':
            return CamVid.CamVid('datasets/CamVid/', mode)
        else:
            raise TypeError('unsupported CamVid mode {}'.format(mode))

    elif dname == 'PascalVoc2012':
        if mode == 'train' or mode == 'val' or mode == 'trainval':
            return PascalVoc.PascalVOC('datasets/VOCdevkit/VOC2012/', mode)

