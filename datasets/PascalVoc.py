#!/usr/bin/python
# -*- encoding: utf-8 -*-



import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os



class PascalVOC(Dataset):
    def __init__(self, root_pth, mode = 'train'):
        super(PascalVOC, self).__init__()
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
            raise ValueError('unrecognized mode: {}'.format(self.mode))

        with open(self.img_map, 'r') as fr:
            self.img_names = fr.read().splitlines()


    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_pth = os.path.join(self.jpg_pth, img_name + '.jpg')
        lb_pth = os.path.join(self.label_pth, img_name + '.png')
        img = cv2.imread(img_pth)
        label = cv2.imread(lb_pth)
        return img, label



if __name__ == '__main__':
    ds = PascalVOC(root_pth = './VOCdevkit/VOC2012/', mode = 'val')
    dl = DataLoader(ds, batch_size = 4, shuffle = True, num_workers = 4)
    im, lb = ds[14]
    #  cv2.imshow('img', im)
    #  cv2.imshow('lb', lb)
    #  cv2.waitKey(0)
    for im, lb in ds:
        pass

