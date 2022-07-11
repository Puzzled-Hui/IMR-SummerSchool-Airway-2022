# -*- coding: utf-8 -*-

'''
Program :   The dataloader for the airway dataset, small dataset from the BAS dataset.
Author  :   Minghui Zhang, Institute of Medical Robotics, Shanghai Jiao Tong University.
File    :   airway_dataset.py
Date    :   2022/7/11 13:51
Version :   V1.0
'''

import os
import random
import numpy as np
import torch
from natsort import natsorted
from copy import deepcopy
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat

from monai.transforms import (
    RandCropByLabelClassesd,
    SpatialCropd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    Compose,
    LabelFilter,
    AddChanneld,
    RandSpatialCropd,
    RandFlipd,
    Orientationd,
    RandRotated,
    ToTensord
)
from monai.utils import set_determinism

from .base_dataset import BaseDataset
from main_code.util import utils


class AirwayDataset(BaseDataset):
    '''
        AirwayDataset : For Pumonary Airway Segmentation
    '''

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--depth', type=int, default=64, help='height used in the crop size')
        parser.add_argument('--height', type=int, default=64, help='height used in the crop size')
        parser.add_argument('--width', type=int, default=64, help='width used in the crop size')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.isTrain
        self.file_rootdir = utils.getabspath([opt.dataroot, opt.phase])
        self.file_list = os.listdir(self.file_rootdir)
        self.file_list = natsorted(self.file_list)

        set_determinism(seed=777)

        self.train_transform = Compose(
            [
                AddChanneld(
                    keys=["image"]),
                # Normalize the function from the 0-255 to 0-1.
                ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),

                # Foreground Crop Prob is set to 0.8
                RandCropByLabelClassesd(keys=["image", "mask"], label_key="mask",
                                        spatial_size=[opt.depth, opt.height, opt.width],
                                        ratios=[0.2, 0.8], num_classes=2, num_samples=1),

                # TODO YOU CAN DO THE EXTRA DATA AUGMENTATION IN THIS PART.

                # Random Rotation in MOANI use radians not angles 0.174 rad = 10°
                RandRotated(keys=["image", "mask"], prob=1.0, range_x=(-0.174, 0.174), range_y=(-0.174, 0.174),
                            range_z=(-0.174, 0.174), mode=["bilinear", "nearest"]),
                # To Tensord
                ToTensord(keys=["image", "mask"], dtype=torch.float)
            ]
        )
        self.test_transform = Compose(
            [
                AddChanneld(
                    keys=["image"]),
                ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                # To Tensord
                ToTensord(keys=["image", "mask"], dtype=torch.float)
            ]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        casename = self.file_list[idx]

        image_path = utils.getabspath([self.file_rootdir, casename, 'processed', casename + '_clean.nii'])
        image_array, origin, spacing = utils.load_itk_image(image_path)
        #print(np.max(image_array), np.min(image_array))
        mask_path = utils.getabspath([self.file_rootdir, casename, 'processed', casename + '_label.nii'])
        label_foreground_array, _, _ = utils.load_itk_image(mask_path)
        label_background_array = 1.0 - label_foreground_array
        mask_list = [label_background_array, label_foreground_array]
        mask_array = np.asarray(mask_list, dtype=np.uint8)

        metadata_dict = {
            'image': image_array,
            'mask': mask_array
        }

        if self.isTrain:
            metadata_dict = self.train_transform(metadata_dict)
        else:
            metadata_dict = self.test_transform(metadata_dict)

        #print(torch.max(metadata_dict[0]['image']), torch.min(metadata_dict[0]['image']))

        if isinstance(metadata_dict, list) and isinstance(metadata_dict[0], dict):
            return (metadata_dict[0])
        elif isinstance(metadata_dict, dict):
            return (metadata_dict)
        else:
            raise TypeError('meta data must be a list[dict] or dict')
