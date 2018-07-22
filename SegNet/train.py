#!/usr/bin/python
# -*- encoding: utf-8 -*-



import torch
import json



with open('./config.json', 'r') as jr:
    cfg = json.load(jr)

print(cfg)
