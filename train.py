#!/usr/bin/python
# -*- encoding: utf-8 -*-


from datasets import get_datasets
from utils.AttrDict import AttrDict
from config import load_cfg
from torch.utils.data import DataLoader
from models import get_model

import torch.nn as nn
import torch
import os
import sys
import logging
import pickle
import numpy as np
import cv2


## TODO:
# 1. image_siza  -- done
# 2. optimizer class encapsuling optimizer and scheduler
# 3. encapsule load checkpoint and model to a method



def resize_label(label, size):
    if label.shape[-2] == size[0] and label.shape[-1] == size[1]:
        return label
    size = (label.shape[0], size[0], size[1])
    new_label = np.empty(size, dtype = np.int32)
    for i, lb in enumerate(label.numpy()):
        new_label[i, ...] = cv2.resize(lb, size[1:], interpolation = cv2.INTER_NEAREST)
    return torch.from_numpy(new_label).long()


def train(cfg_file):
    ## config
    cfg = load_cfg(cfg_file)

    ## logging
    FORMAT = '%(levelname)s %(filename)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    ## data loader
    trainset = get_datasets(cfg.train.datasets, 'train')
    valset = get_datasets(cfg.val.datasets, 'val')
    trainloader = DataLoader(trainset,
                            batch_size = cfg.train.batch_size,
                            shuffle = True,
                            num_workers = 4,
                            drop_last = True)
    valloader = DataLoader(valset,
                            batch_size = 4,
                            shuffle = True,
                            num_workers = 4,
                            drop_last = True)

    ## network and checkpoint
    model = get_model(cfg).float().cuda()
    print(model)
    if cfg.model.class_weight is not None:
        weight = torch.Tensor(cfg.model.class_weight)  # ignore some labels or set weight
    else:
        weight = None
    Loss = nn.CrossEntropyLoss(weight = weight).cuda()


    ## optimizer
    optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr = cfg.optimizer.base_lr,
                    momentum = cfg.optimizer.momentum,
                    weight_decay = cfg.optimizer.weight_decay)
    if cfg.optimizer.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size = cfg.optimizer.stepsize,
                        gamma = cfg.optimizer.gamma)

    ## checkpoint
    save_path = cfg.train.out_path
    if not os.path.exists(save_path): os.makedirs(save_path)
    save_name = os.path.join(save_path, 'model.pytorch')
    if os.path.exists(save_name): return
    model_ckpts = os.listdir(save_path)
    its = [int(os.path.splitext(el)[0].split('_')[2]) for el in model_ckpts if el[:5] == 'model']
    start_it = 0
    if len(its) > 0:
        start_it = max(its)
        model_ckpt = os.path.join(save_path, ''.join(['model_iter_', str(start_it), '.pytorch']))
        logger.info('resume from checkpoint: {}\n'.format(model_ckpt))
        model.load_state_dict(torch.load(model_ckpt))
        optim_checkpoint = model_ckpt.replace('model', 'optim')
        optimizer.load_state_dict(torch.load(optim_checkpoint))

    ## multi-gpu
    model = nn.DataParallel(model, device_ids = None)

    ## train
    result = AttrDict({
        'train_loss': [],
        'val_loss': [],
        'cfg': cfg
        })
    trainiter = iter(trainloader)
    for it in range(start_it, cfg.train.max_iter):
        model.train()
        optimizer.zero_grad()
        try:
            im, label = next(trainiter)
            if not im.shape[0] == cfg.train.batch_size: continue
        except StopIteration:
            trainiter = iter(trainloader)
            im, label = next(trainiter)

        im = im.cuda().float()
        num_class = cfg.model.num_class
        logits = model(im)
        label = resize_label(label, logits.shape[2:])
        label = label.cuda().long().contiguous().view(-1, )
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, num_class)
        loss = Loss(logits, label)
        loss_value = loss.detach().cpu().numpy()
        result.train_loss.append(loss_value)
        loss.backward()

        optimizer.step()
        if cfg.optimizer.lr_policy == 'step':
            scheduler.step()

        if it == 0: continue
        if it % 20 == 0:
            logger.info('iter: {}/{}, loss: {}'.format(it, cfg.train.max_iter, loss_value))
        if it % cfg.val.valid_iter == 0:
            valid_loss, acc_clss, acc_all = val_one_epoch(model, Loss, valloader, cfg)
            result.val_loss.append(valid_loss)
        if it % cfg.train.snapshot_iter == 0:
            save_model_name = os.path.join(save_path, ''.join(['model_iter_', str(it), '.pytorch']))
            save_optim_name = save_model_name.replace('model', 'optim')
            logger.info('saving snapshot to: {}'.format(save_path))
            torch.save(model.module.state_dict(), save_model_name)
            torch.save(optimizer.state_dict(), save_optim_name)

    logger.info('training done')
    save_name = os.path.join(save_path, 'model.pytorch')
    logger.info('saving model to: {}'.format(save_name))
    model.cpu()
    torch.save(model.module.state_dict(), save_name)
    with open(save_path + '/result.pkl', 'wb') as fw:
        pickle.dump(result, fw)
    while True:
        try:
            im, label = next(trainiter)
        except StopIteration:
            break
    print('everything done')


def val_one_epoch(model, Loss, valid_loader, cfg):
    model.eval()
    val_loss = []
    acc_list_list = []
    acc_list = [[] for i in range(cfg.model.num_class)]
    for img, label in valid_loader:
        logits = model(img.cuda().float())
        label = resize_label(label, logits.shape[2:])
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, cfg.model.num_class)
        label = label.cuda().long().contiguous().view(-1, )
        loss = Loss(logits, label)
        val_loss.append(loss.detach().cpu().numpy())

        clsses = logits.detach().cpu().numpy().argmax(axis = 1)
        lbs = label.cpu().numpy().astype(np.int64)
        for idx in range(cfg.model.num_class - 1):
            indices = np.where(lbs == idx)
            clss = clsses[indices]
            acc_list[idx].extend(list(clss == idx))
    valid_loss = sum(val_loss) / len(val_loss)
    acc_per_class = np.array([sum(el) / len(el) if not len(el) == 0 else None for el in acc_list])
    acc_list_all = []
    [acc_list_all.extend(el) for el in acc_list]
    acc_all = sum(acc_list_all) / len(acc_list_all)

    print('=======================================')
    print('result on validation set:')
    print('accuracy per class:\n {}'.format(acc_per_class.reshape(-1, 1)))
    print('accuracy all: {}'.format(acc_all))
    print('validation loss: {}'.format(valid_loss))
    print('=======================================')

    return valid_loss, acc_per_class, acc_all



if __name__ == '__main__':
    train()
