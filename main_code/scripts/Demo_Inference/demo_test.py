# -*- coding: utf-8 -*-

'''
Program :   Demo test code for the airway prediction
Author  :   Minghui Zhang, sjtu
File    :   test.py
Date    :   2022/07/11 14:00
Version :   V1.0
'''

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))
import argparse
import time
import numpy as np
import torch
import SimpleITK as sitk
from sklearn.metrics import precision_score, recall_score

from model_baseline import UNet3D
from main_code.util.utils import dice_coef_np


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = list(reversed(itkimage.GetOrigin()))
    numpySpacing = list(reversed(itkimage.GetSpacing()))
    return numpyImage, numpyOrigin, numpySpacing


def save_itk(image, origin, spacing, filename):
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)


def normalize_CT(image: np.ndarray) -> np.ndarray:
    min = np.min(image)
    max = np.max(image)
    image_normalized = (image - min) / (max - min)
    return image_normalized


def inference(model, x, step, cube_size, threshold):
    model.eval()
    pred = torch.from_numpy(np.zeros(shape=(x.shape[0], 2, x.shape[2], x.shape[3], x.shape[4])))
    pred_num = torch.from_numpy(np.zeros(shape=(x.shape[0], 2, x.shape[2], x.shape[3], x.shape[4])))
    xnum = (x.shape[2] - cube_size[0]) // step + 1 \
        if (x.shape[2] - cube_size[0]) % step == 0 else (x.shape[2] - cube_size[0]) // step + 2
    ynum = (x.shape[3] - cube_size[1]) // step + 1 \
        if (x.shape[3] - cube_size[1]) % step == 0 else (x.shape[3] - cube_size[1]) // step + 2
    znum = (x.shape[4] - cube_size[2]) // step + 1 \
        if (x.shape[4] - cube_size[2]) % step == 0 else (x.shape[4] - cube_size[2]) // step + 2
    for xx in range(xnum):
        xl = step * xx
        xr = step * xx + cube_size[0]
        if xr > x.shape[2]:
            xr = x.shape[2]
            xl = x.shape[2] - cube_size[0]
        for yy in range(ynum):
            yl = step * yy
            yr = step * yy + cube_size[1]
            if yr > x.shape[3]:
                yr = x.shape[3]
                yl = x.shape[3] - cube_size[1]
            for zz in range(znum):
                zl = step * zz
                zr = step * zz + cube_size[2]
                if zr > x.shape[4]:
                    zr = x.shape[4]
                    zl = x.shape[4] - cube_size[2]
                x_input = x[:, :, xl:xr, yl:yr, zl:zr]
                p = model(x_input).detach().cpu()
                pred[:, :, xl:xr, yl:yr, zl:zr] += p
                pred_num[:, :, xl:xr, yl:yr, zl:zr] += 1
                torch.cuda.empty_cache()
    pred = pred / pred_num
    pred_airway = np.asarray(pred[0, 1, ...].numpy() > threshold, dtype=np.uint8)
    return pred_airway


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default='image.nii', help="input_path")
parser.add_argument("--label_path", type=str, default='label.nii', help="label_path")
parser.add_argument("--output_path", type=str, default='pred.nii.gz', help="output_path")

if __name__ == "__main__":
    args = parser.parse_args()
    with torch.no_grad():
        # devices
        device_ids = [0]
        device = []
        for (index, gpu_id) in enumerate(device_ids):
            device.append(torch.device('cuda:{}'.format(gpu_id)) if device_ids else torch.device('cpu'))
        # weights
        weight_path = 'weight.pth'
        state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda(0))
        # initilize the model and load the weight.
        model = UNet3D(in_channels=1, out_channels=2, finalsigmoid=1, fmaps_degree=16, GroupNormNumber=2,
                       fmaps_layer_number=4, layer_order='cpi', device=device)
        model.load_state_dict(state_dict)

        # demo image / label
        input_path = args.input_path
        output_path = args.output_path
        label_path = args.label_path
        image, origin, spacing = load_itk_image(input_path)
        label, _, _ = load_itk_image(label_path)
        label = label.astype(np.uint8)
        image_normalized = normalize_CT(image)
        image_tensor = torch.from_numpy(image_normalized).float().unsqueeze(0).unsqueeze(0).to(device[0])
        pred_airway = inference(model=model, x=image_tensor, step=64, cube_size=[128, 128, 128], threshold=0.5)
        dice_coefficient = dice_coef_np(label, pred_airway)
        precision_coefficient = precision_score(label.flatten(), pred_airway.flatten())
        recall_coefficient = recall_score(label.flatten(), pred_airway.flatten())
        print('pred successfully!')
        print('Dice:{:.4f} Precision:{:.4f} Recall:{:.4f}'.format(dice_coefficient, precision_coefficient,
                                                                  recall_coefficient))
        save_itk(pred_airway, origin, spacing, output_path)
