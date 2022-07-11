# -*- coding: utf-8 -*-

'''
Program :   UNet3D model.
Author  :   Minghui Zhang, Institute of Medical Robotics, Shanghai Jiao Tong University.
File    :   unet3d_model.py
Date    :   2022/7/11 15:05
Version :   V1.0
'''

import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks

from main_code.util.losses import Dice_with_Focal, DiceLoss


class Unet3dModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add new dataset-specific options, and rewrite default values for existing options.
        """
        return parser

    ''' ================================================= Core Class Methods ======================================================='''

    def __init__(self, opt):
        """
            Initialize the model.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['Unet3d_final_output']
        self.model_names = ['Unet3d']
        # specify the choice of loss function
        self.loss_mode = opt.loss_mode
        # define the networks, only support single GPU in this version. Easy to extend to the multi-gpus via torch.nn.Parallel
        if (len(self.device) == 1):
            self.device = self.device[0]
        else:
            raise ValueError('GPU Device > 0, Need Modification!')

        self.netUnet3d = networks.define_Unet3d(in_channels=opt.in_channels, out_channels=opt.out_channels,
                                                finalsigmoid=opt.final_sigmoid, fmaps_degree=opt.init_fmaps_degree,
                                                GroupNormNumber=4, fmaps_layer_number=opt.fmaps_layer_number,
                                                layer_order=opt.layer_order, device=self.device)

        if self.isTrain:
            if (opt.loss_mode == 'Dice'):
                self.criterionUnet3d_final_output = DiceLoss(sigmoid_normalization=opt.final_sigmoid, ignore_index=None)

            # TODO YOU CAN EXTEND OTHER LOSS FUNCTION HERE.

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if (opt.optim_mode == 'Adam'):
                self.optimizer_Unet3d = torch.optim.Adam(params=self.netUnet3d.parameters(), lr=opt.init_lr,
                                                         betas=(opt.adam_beta1, 0.999))
            elif (opt.optim_mode == 'SGD'):
                self.optimizer_Unet3d = torch.optim.SGD(params=self.netUnet3d.parameters(), lr=opt.init_lr,
                                                        momentum=opt.sgd_momentum,
                                                        weight_decay=opt.sgd_weight_decay)
            else:
                raise NameError(opt.optim_mode + 'is not specified!')
            self.optimizers.append(self.optimizer_Unet3d)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (tuple): include the data itself and its metadata information.

        """
        self.image = input['image'].to(self.device)
        self.mask = input['mask'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.prediction = self.netUnet3d(self.image)

    def backward(self):
        self.loss_Unet3d_final_output = self.criterionUnet3d_final_output(self.prediction, self.mask)
        self.loss_Unet3d_final_output.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_Unet3d.zero_grad()
        self.backward()
        self.optimizer_Unet3d.step()

    ''' ================================================= Core Class Methods ======================================================='''

    """ ======================================================= Other Methods ======================================================="""

    def fetch_model(self, name):
        return getattr(self, 'net' + name)

    """ ======================================================= Other Methods ======================================================="""
