#!/usr/bin/python
# -*- encoding: utf-8 -*-



import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
import cv2
import os


'''
Pascal VOC Segmentation consists of 20 classes. The label array can be created py PIL.Image:
    label = np.array(PIL.Image.open(label_img_pth))
In the label numpy array, 0 stands for background class and 255 stands for the boundary of an object or class region, while the other numbers from 1 to 20 stands for the 20 classes.
'''


class PascalVOC(Dataset):
    def __init__(self, root_pth, mode = 'train', *args, **kwargs):
        super(PascalVOC, self).__init__(*args, **kwargs)
        self.mode = mode
        self.jpg_pth = os.path.join(root_pth, 'JPEGImages')
        self.label_pth = os.path.join(root_pth, 'SegmentationClass')

        if self.mode == 'train':
            self.img_map = os.path.join(root_pth, 'ImageSets/Segmentation', 'train.txt')
        elif self.mode == 'val':
            self.img_map = os.path.join(root_pth, 'ImageSets/Segmentation', 'val.txt')
        elif self.mode == 'trainval':
            self.img_map = os.path.join(root_pth, 'ImageSets/Segmentation', 'trainval.txt')
        else:
            raise ValueError('unrecognized Pascal Voc mode: {}'.format(self.mode))

        with open(self.img_map, 'r') as fr:
            self.img_names = fr.read().splitlines()


    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_pth = os.path.join(self.jpg_pth, img_name + '.jpg')
        lb_pth = os.path.join(self.label_pth, img_name + '.png')
        img = cv2.imread(img_pth)
        img = cv2.resize(img, (321, 321)).astype(np.float)
        img[:, :, 0] -= 104.008
        img[:, :, 1] -= 122.675
        img[:, :, 2] -= 116.669
        flip_p = random.uniform(0, 1)
        if flip_p > 0.5:
            img = np.fliplr(img).copy()
        img = img.transpose(2, 0, 1)
        label = np.array(Image.open(lb_pth))
        label[label==255] = 21 # set the boundary to be category 21 (ignored label)
        label = cv2.resize(label, (321, 321),
                interpolation = cv2.INTER_NEAREST).astype(np.float)
        return img, label



if __name__ == '__main__':
    ds = PascalVOC(root_pth = './VOCdevkit/VOC2012/', mode = 'train')
    dl = DataLoader(ds, batch_size = 4, shuffle = True, num_workers = 1)
    im, lb = ds[14]
    print(im.shape)
    print(lb.shape)
    lb[lb==0] = 255
    #  cv2.imshow('img', im)
    #  cv2.imshow('lb', lb)
    #  cv2.waitKey(0)
    #  for im, lb in ds:
    #      print(im[0, 0, 1])
    #      pass
    trainiter = iter(dl)
    im, label = next(trainiter)
    #  for i in range(200):
    #      im, label = next(trainiter)


