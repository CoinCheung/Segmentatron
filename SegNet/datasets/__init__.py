#!/usr/bin/python
# -*- encoding: utf-8 -*-


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

