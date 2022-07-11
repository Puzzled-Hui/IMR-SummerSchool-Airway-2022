# -*- coding: utf-8 -*-

'''
Program :   1. utils for the networks modules
Author  :   Minghui Zhang, sjtu
File    :   util.py
Date    :   2022/1/13 13:28
Version :   V1.0
'''

import torch.nn as nn
import torch.nn.functional as F
import torch

##############################################################################
# Different Activation Function
##############################################################################
class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self,x):
        x = x * torch.sigmoid(x)
        return x


class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self,x):
        x = x * torch.tanh(F.softplus(x))
        return x
