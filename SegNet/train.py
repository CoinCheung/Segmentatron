#!/usr/bin/python
# -*- encoding: utf-8 -*-


from datasets import get_datasets
from model import SegNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

import json
import os


'''
    1. 训练过程中val
    2. python infer.py example.jpg
    3. accuracy
'''


def train():
    ## config
    with open('./config.json', 'r') as jr:
        cfg = json.load(jr)
    opt = cfg['optimizer']
    trn = cfg['train']

    ## data loader
    trainset = get_datasets(trn['datasets'], 'train')
    valset = get_datasets(trn['datasets'], 'val')
    trainloader = DataLoader(trainset,
                            batch_size = trn['batch_size'],
                            shuffle = True,
                            num_workers = 4)
    valloader = DataLoader(valset,
                            batch_size = 4,
                            shuffle = True,
                            num_workers = 4)

    ## network and loss
    segnet = SegNet().float().cuda()
    segnet = nn.DataParallel(segnet, device_ids = None)
    weight = torch.Tensor([0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418, 0.6823, 6.2478, 7.3614, 0])  # ignore label 11
    Loss = nn.CrossEntropyLoss(weight = weight).cuda()

    ## optimizer
    opt = cfg['optimizer']
    optimizer = torch.optim.SGD(
                    segnet.parameters(),
                    lr = opt['base_lr'],
                    momentum = opt['momentum'],
                    weight_decay = opt['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size = opt['stepsize'],
                    gamma = opt['gamma'])

    ## checkpoint
    save_path = trn['out_path']
    if not os.path.exists(save_path): os.makedirs(save_path)
    save_name = os.path.join(save_path, 'model.pytorch')
    if os.path.exists(save_name): return
    models = os.listdir(save_path)
    its = [int(os.splitext(el)[1].split('_')[2]) for el in models if el[:5] == 'model']
    if len(its) > 0:
        max_it = max(its)
        checkpoint = os.path.join(save_path, ''.join(['model_iter_', str(max_it), '.pytorch']))
        segnet.load_state_dict(torch.load(checkpoint))

    ## train
    segnet.train()
    epoch = trn['max_iter'] // len(trainset) + 1
    it = 0
    for e in range(epoch):
        for im, label in trainloader:
            im = im.cuda().float()
            label = label.cuda().long().contiguous().view(-1, )

            logits = segnet(im).permute(0, 2, 3, 1).contiguous().view(-1, 12)

            loss = Loss(logits, label)
            segnet.zero_grad()
            loss.backward()
            scheduler.step()
            loss_val = loss.detach().cpu().numpy()

            it += 1
            if it % 20 == 0:
                print('iter: {}/{}, loss: {}'.format(it, trn['snapshot'], loss_val))
            if it % trn['snapshot'] == 0:
                save_name = os.path.join(save_path, ''.join(['model_iter_', str(it), '.pytorch']))
                torch.save(segnet.state_dict(), save_name)
            if it == trn['max_iter']:
                save_name = os.path.join(save_path, 'model.pytorch')
                torch.save(segnet.state_dict(), save_name)


if __name__ == '__main__':
    train()
