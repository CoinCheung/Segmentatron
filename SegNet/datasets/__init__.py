#!/usr/bin/python
# -*- encoding: utf-8 -*-


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import CamVid


dataset_names = ['CamVid',]


def get_datasets(dname, mode):
    assert dname in dataset_names, 'unknown dataset: {}'.format(dname)
    if mode == 'train':
        return CamVid.get_CamVid_train()
    elif mode == 'val':
        return CamVid.get_CamVid_val()
    elif mode == 'test':
        return CamVid.get_CamVid_test()
    else:
        raise TypeError('unsupported mode {}'.format(mode))

