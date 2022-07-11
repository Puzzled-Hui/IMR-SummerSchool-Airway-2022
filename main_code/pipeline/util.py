# -*- coding: utf-8 -*-

'''
Program :   1. Include some useful utils for the pipeline of train, debug and inference
Author  :   Minghui Zhang, sjtu
File    :   util.py
Date    :   2022/1/15 14:20
Version :   V1.0
'''

import random
import numpy as np
import torch





def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False