#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch.nn as nn
import torch
import os
import sys
import logging


FORMAT = '%(levelname)s %(filename)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


class Optimizer(object):
    def __init__(self, param_list, cfg):
        self.cfg = cfg
        self.use_scheduler = True
        if cfg.optimizer.lr_policy == 'step': self.use_scheduler = True
        self.optimizer = torch.optim.SGD(
                        param_list,
                        lr = self.cfg.optimizer.base_lr,
                        momentum = self.cfg.optimizer.momentum,
                        weight_decay = self.cfg.optimizer.weight_decay)

        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                            self.optimizer,
                            step_size = self.cfg.optimizer.stepsize,
                            gamma = self.cfg.optimizer.gamma)


    def step(self):
        self.optimizer.step()
        if self.use_scheduler:
            self.scheduler.step()


    def zero_grad(self):
        self.optimizer.zero_grad()


    def load_checkpoint(self, save_path, iter_num = None):
        if self.use_scheduler and iter_num is None:
            raise NameError('Need assign iter number')
        optim_name = ''.join(['optim_iter_', str(iter_num)])
        optim_path = os.path.join(save_path, optim_name)
        logger.info('loading optimizer state dict')
        self.optimizer.load_state_dict(torch.load(optim_path))
        if self.use_scheduler:
            schdlr_name = optim_name.replace('optim', 'scheduler')
            schdlr_path = os.path.join(save_path, schdlr_name)
            logger.info('loading scheduler state dict')
            self.scheduler.load_state_dict(torch.load(schdlr_path))


    def save_checkpoint(self, save_path, iter_num = None):
        if iter_num is not None:
            optim_name = ''.join(['optim_iter_', str(iter_num)])
            schdlr_name = optim_name.replace('optim', 'scheduler')
        else:
            raise NameError('Snapshot iter number should be assigned ')
        optim_path = os.path.join(save_path, optim_name)
        logger.info('saving optimizer state dict')
        torch.save(self.optimizer.state_dict(), optim_path)
        if self.use_scheduler:
            schdlr_path = os.path.join(save_path, schdlr_name)
            logger.info('saving scheduler state dict')
            torch.save(self.scheduler.state_dict(), schdlr_path)
