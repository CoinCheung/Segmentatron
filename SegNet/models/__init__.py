#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from SegNet import *


def get_model(model_name):
    if model_name == "SegNet":
        return SegNet()

    else:
        raise NameError("unsupported model type {}".format(model_name))
