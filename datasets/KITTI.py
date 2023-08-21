# KITTI.py
# Implements KITTi dataloader.
# Adapted from the dataset used for AdaBins (https://github.com/shariqfarooq123/AdaBins)

import os, sys, logging

import numpy as np
import random
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets.DepthDataset import DepthDataset
import misc_utils

class KITTI(DepthDataset):
    def __init__(self, args, mode, transform=None):
        super().__init__(args)
        self.args = args

        self.mode = mode
        self.transform = transform
        self.logger = logging.getLogger(f"{__name__}[{self.mode}]")

        self.logger.info(f"Creating dataset (mode: {self.mode})")

        assert self.mode in ['train', 'online_eval', 'test'], f"Error: Dataset mode {self.mode} not recognised."

        # NYU-specific normalisation factors (see DepthDataset normalisation methods)
        self.image_norm_factor = 255.0
        self.depth_norm_factor = 1000.0

        # Set up file paths
        self.kitti_base_path = os.path.join(args.paths.data_dir, args.kitti.base_path)
        self.data_path = os.path.join(self.kitti_base_path, self.args.kitti.data_path)
        self.gt_path = os.path.join(self.kitti_base_path, self.args.kitti.gt_path)

        # Load data filenames into memory, set per-split info (e.g. paths)
        if self.mode == 'online_eval':
            self.logger.debug(f"Filenames file: {self.args.kitti.filenames_file_eval}")
            with open(self.args.kitti.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
            
        else:
            self.logger.debug(f"Filenames file: {self.args.kitti.filenames_file_train}")
            with open(self.args.kitti.filenames_file_train, 'r') as f:
                self.filenames = f.readlines()


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx):
        if idx >= len(self.filenames):
            raise StopIteration
        
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])
        if self.mode == 'train' and self.args.kitti.use_right is True and random.random() > 0.5:
            image_path = os.path.join(self.data_path, misc_utils.remove_leading_slash(sample_path.split()[3]))
            depth_path = os.path.join(self.gt_path, misc_utils.remove_leading_slash(sample_path.split()[4]))
        else:
            image_path = os.path.join(self.data_path, misc_utils.remove_leading_slash(sample_path.split()[0]))
            depth_path = os.path.join(self.gt_path, misc_utils.remove_leading_slash(sample_path.split()[1]))

        has_valid_depth = False

        # Load files from disk
        image = Image.open(image_path)
        try:
            depth_gt = Image.open(depth_path)
            has_valid_depth = True
        except IOError:
            depth_gt = None
            self.logger.debug(f"Missing depth GT for {image_path}")

        # Skip bad examples - delete the path from the filenames list and try again.
        if not has_valid_depth:
            del self.filenames[idx]
            return self.__getitem__(idx)
        
        sample = {
            'image': image,
            'depth_gt': depth_gt,
            'focal': focal,
            'has_valid_depth': has_valid_depth,
            'image_path': image_path,
            'depth_path': depth_path
            }
        

        if self.transform:
            sample = self.transform(sample)

        return sample
        
            
