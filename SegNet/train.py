#!/usr/bin/python
# -*- encoding: utf-8 -*-


from datasets import get_datasets
from utils.AttrDict import AttrDict
from config import load_cfg
from model import SegNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

import os
import sys
import logging
#  import Piclkle as pickle
import pickle


'''
    2. python infer.py example.jpg
    3. accuracy, classification accuracy, with label 11 ignored (0-11 valid)
'''


def train():
    ## config
    cfg = load_cfg('./config.json')

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

    ## network and loss
    segnet = SegNet().float().cuda()
    segnet = nn.DataParallel(segnet, device_ids = None)
    weight = torch.Tensor([0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418, 0.6823, 6.2478, 7.3614, 0])  # ignore label 11
    Loss = nn.CrossEntropyLoss(weight = weight).cuda()

    ## optimizer
    optimizer = torch.optim.SGD(
                    segnet.parameters(),
                    lr = cfg.optimizer.base_lr,
                    momentum = cfg.optimizer.momentum,
                    weight_decay = cfg.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size = cfg.optimizer.stepsize,
                    gamma = cfg.optimizer.gamma)

    ## checkpoint
    save_path = cfg.train.out_path
    if not os.path.exists(save_path): os.makedirs(save_path)
    save_name = os.path.join(save_path, 'model.pytorch')
    if os.path.exists(save_name): return
    models = os.listdir(save_path)
    its = [int(os.path.splitext(el)[0].split('_')[2]) for el in models if el[:5] == 'model']
    it = 0
    if len(its) > 0:
        it = max(its)
        model_checkpoint = os.path.join(save_path, ''.join(['model_iter_', str(it), '.pytorch']))
        optim_checkpoint = model_checkpoint.replace('model', 'optim')
        logger.info('resume from checkpoint: {}\n'.format(model_checkpoint))
        segnet.load_state_dict(torch.load(model_checkpoint))
        optimizer.load_state_dict(torch.load(optim_checkpoint))

    ## train
    epoch = int((cfg.train.max_iter - it) // (len(trainset) / cfg.train.batch_size) + 1)
    result = AttrDict({
        'train_loss': [],
        'val_loss': [],
        'cfg': cfg
        })
    for e in range(epoch):
        for im, label in trainloader:
            im = im.cuda().float()
            label = label.cuda().long().contiguous().view(-1, )

            segnet.train()
            optimizer.zero_grad()
            logits = segnet(im).permute(0, 2, 3, 1).contiguous().view(-1, 12)
            loss = Loss(logits, label)
            loss_value = loss.detach().cpu().numpy()
            result.train_loss.append(loss_value)
            loss.backward()

            scheduler.step()
            optimizer.step()

            it += 1
            if it % 20 == 0:
                logger.info('iter: {}/{}, loss: {}'.format(it, cfg.train.max_iter, loss_value))
            if it % cfg.train.snapshot == 0:
                save_model_name = os.path.join(save_path, ''.join(['model_iter_', str(it), '.pytorch']))
                save_optim_name = save_model_name.replace('model', 'optim')
                logger.info('saving snapshot to: {}'.format(save_path))
                torch.save(segnet.state_dict(), save_model_name)
                torch.save(optimizer.state_dict(), save_optim_name)
            if it == cfg.train.max_iter:
                logger.info('training done')
                save_name = os.path.join(save_path, 'model.pytorch')
                logger.info('saving model to: {}'.format(save_name))
                torch.save(segnet.state_dict(), save_name)
                with open('./loss.pkl', 'wb') as fw:
                    pickle.dump(result, fw)
                print('everything done')
                return
        valid_loss = val_one_epoch(segnet, Loss, valloader)
        result.val_loss.append(valid_loss)
        logger.info('epoch {}, validation loss: {}'.format(e, valid_loss))


def val_one_epoch(model, Loss, valid_loader):
    val_loss = []
    model.eval()
    for img, label in valid_loader:
        logits = model(img.cuda().float()).permute(0, 2, 3, 1).contiguous().view(-1, 12)
        label = label.cuda().long().contiguous().view(-1, )
        loss = Loss(logits, label)
        val_loss.append(loss.detach().cpu().numpy())
    return sum(val_loss) / len(val_loss)



if __name__ == '__main__':
    train()
