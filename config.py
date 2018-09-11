#!/usr/bin/python
# -*- encoding: utf-8 -*-

from utils.AttrDict import AttrDict
import json

cfg = AttrDict()
cfg.model = AttrDict()
cfg.optimizer = AttrDict()
cfg.train = AttrDict()
cfg.test = AttrDict()


def load_cfg(jfile):
    with open(jfile, 'r') as jr:
        jobj = json.load(jr)
    cfg.model = AttrDict(jobj['model'])
    cfg.optimizer = AttrDict(jobj['optimizer'])
    cfg.train = AttrDict(jobj['train'])
    cfg.test = AttrDict(jobj['test'])
    cfg.val = AttrDict(jobj['val'])
    cfg.out_path = jobj['out_path']

    return cfg
