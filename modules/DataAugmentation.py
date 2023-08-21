# DataAugmentation.py
# nn.Module to perform the various data augmentation steps needed for training.

import os, sys, logging
from types import NoneType
from typing import Dict

from PIL import Image
import numpy as np
import kornia
import torch
import torch.nn as nn
from torchvision import transforms
import kornia
import matplotlib.pyplot as plt


class DataAugmentation(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.imgsz = self.args[self.args.basic.dataset].dimensions_train
        self.random_crop = kornia.augmentation.RandomCrop(size=(self.imgsz[0], self.imgsz[1]))
        self.random_rotate_bilinear = kornia.augmentation.RandomRotation(degrees=self.args[self.args.basic.dataset].degree, resample='bilinear')
        self.random_rotate_nearest = kornia.augmentation.RandomRotation(degrees=self.args[self.args.basic.dataset].degree, resample='nearest')
        self.random_horizontal_flip = kornia.augmentation.RandomHorizontalFlip(p=0.5)
        self.random_planckian_jitter = kornia.augmentation.RandomPlanckianJitter(mode='blackbody', p=0.5)


    def show_imdepth(self, batch):
        # plt.imshow(batch['image'][0].permute(1, 2, 0).cpu().numpy())
        # plt.show()
        plt.imshow(batch['depth_gt'][0][0].cpu().numpy())
        plt.show()


    def forward(self, batch):
        """Perform data augmentation. Expects the input images to be in range [0, 1]. Doesn't perform normalisation.

        Args:
            batch (dict of "image" and "depth", and maybe others): the batch dict
            device : The current device of this module. Set by the pl.LightningModule and used for RNG.
        
        Returns:
            batch (dict of "image" and "depth", and others if needed): The (possibly augmented) batch dict.
        """

        # if self.args[self.args.basic.dataset].do_random_rotate is True:
        #     batch['image'] = self.random_rotate_bilinear(batch['image'])
        #     batch['depth_gt'] = self.random_rotate_nearest(batch['depth_gt'], params=self.random_rotate_bilinear._params)

        # # Random crop
        # batch['image'] = self.random_crop(batch['image'])
        # batch['depth_gt'] = self.random_crop(batch['depth_gt'], params=self.random_crop._params)

        # train_preprocess
        # Random flip in the y axis
        batch['image'] = self.random_horizontal_flip(batch['image'])
        batch['depth_gt'] = self.random_horizontal_flip(batch['depth_gt'], params=self.random_horizontal_flip._params)

        # Random gamma augment
        random_gamma = (torch.rand((batch['image'].shape[0], 1, 1, 1), device=batch['image'].device) - 0.5) * 0.2  # Tensor of B random numbers (in range [-0.1, 0.1))
        random_gamma += 1.0  # In range [0.9, 1.1)
        batch['image'] = batch['image'] ** random_gamma

        # Random Planckian Jitter (a replacement for old-style brightness and colour jitter, see arXiv:2202.07993)
        batch['image'] = self.random_planckian_jitter(batch['image'])

        return batch
