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
    model = get_model(cfg.model.backbone).float().cuda()
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
        try:
            im, label = next(trainiter)
            if not im.shape[0] == cfg.train.batch_size: continue
        except StopIteration:
            trainiter = iter(trainloader)
            im, label = next(trainiter)

        im = im.cuda().float()
        label = label.cuda().long().contiguous().view(-1, )

        model.train()
        optimizer.zero_grad()
        logits = model(im).permute(0, 2, 3, 1).contiguous().view(-1, 12)
        loss = Loss(logits, label)
        loss_value = loss.detach().cpu().numpy()
        result.train_loss.append(loss_value)
        loss.backward()

        optimizer.step()
        if cfg.optimizer.lr_policy == 'step':
            scheduler.step()

        it += 1
        if it % 20 == 0:
            logger.info('iter: {}/{}, loss: {}'.format(it, cfg.train.max_iter, loss_value))
        if it % cfg.train.valid_iter == 0:
            valid_loss, acc_clss, acc_all = val_one_epoch(model, Loss, valloader)
            result.val_loss.append(valid_loss)
            print('=======================================')
            logger.info('validation')
            print('accuracy per class:\n {}'.format(acc_clss.reshape(-1, 1)))
            print('accuracy all: {}'.format(acc_all))
            print('loss: {}'.format(valid_loss))
            print('=======================================')
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


def val_one_epoch(model, Loss, valid_loader):
    model.eval()
    val_loss = []
    acc_class_list = []
    acc_list = []
    for img, label in valid_loader:
        logits = model(img.cuda().float()).permute(0, 2, 3, 1).contiguous().view(-1, 12)
        label = label.cuda().long().contiguous().view(-1, )
        loss = Loss(logits, label)
        val_loss.append(loss.detach().cpu().numpy())

        clsses = logits.detach().cpu().numpy().argmax(axis = 1)
        lbs = label.cpu().numpy().astype(np.int64)
        acc = np.mean(lbs == clsses)
        acc_class = []
        for idx in range(11):
            indices = np.where(lbs == idx)
            clss = clsses[indices]
            acc_class.append(np.mean(clss == idx))
        acc_class_list.append(acc_class)
        acc_list.append(np.mean(clsses == lbs))
    acc_mtx = np.array(acc_class_list)
    acc_per_class = np.mean(acc_mtx, axis = 0)
    acc_all = sum(acc_list) / len(acc_list)

    return sum(val_loss) / len(val_loss), acc_per_class, acc_all



if __name__ == '__main__':
    train()
