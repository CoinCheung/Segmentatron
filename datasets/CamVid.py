#!/usr/bin/python
# -*- encoding: utf-8 -*-



import torch
from torch.utils.data import Dataset, DataLoader
import cv2


class CamVid(Dataset):
    '''
    a wrapper of CamVid dataset.
    '''
    def __init__(self, data_path, mode='train'):
        super(CamVid, self).__init__()
        if mode == 'train':
            fns_path = data_path + '/train.txt'
            with open(fns_path, 'r') as fr:
                self.fns = fr.read().splitlines()
                self.imgs = [''.join([data_path, '/train/', el]) for el in self.fns]
                self.labels = [''.join([data_path, '/trainannot/', el]) for el in self.fns]
        elif mode == 'val':
            fns_path = data_path + '/val.txt'
            with open(fns_path, 'r') as fr:
                self.fns = fr.read().splitlines()
                self.imgs = [''.join([data_path, '/val/', el]) for el in self.fns]
                self.labels = [''.join([data_path, '/valannot/', el]) for el in self.fns]
        elif mode == 'test':
            fns_path = data_path + '/test.txt'
            with open(fns_path, 'r') as fr:
                self.fns = fr.read().splitlines()
                self.imgs = [''.join([data_path, '/test/', el]) for el in self.fns]
                self.labels = [''.join([data_path, '/testannot/', el]) for el in self.fns]
        else:
            assert False, 'unknow mode: {}'.format(mode)


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        img = img.transpose(2, 0, 1)
        label = cv2.imread(self.labels[idx])[..., 0]
        #  label = label.transpose(2, 0, 1)
        return img, label


def get_CamVid_train():
    return CamVid('datasets/CamVid/', 'train')


def get_CamVid_val():
    return CamVid('datasets/CamVid/', 'val')


def get_CamVid_test():
    return CamVid('datasets/CamVid/', 'test')


if __name__ == '__main__':
    import os
    print(os.path.abspath('./'))
    dataset = CamVid('./CamVid/', 'train')

    dl = DataLoader(dataset, batch_size = 4, shuffle = True, num_workers = 4)
    for im, label in dl:
        print(type(im))
        print(im.shape)
        label = label * 50
        #  cv2.imshow('img', im[0].numpy())
        #  cv2.imshow('label', label[0].numpy())
        #  cv2.waitKey(0)
    print(len(dataset))


