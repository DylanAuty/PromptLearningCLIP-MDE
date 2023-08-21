# NYUD2.py
# Contains dataset definition for NYUD2
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

class NYUD2(DepthDataset):
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
        self.nyu_base_path = os.path.join(args.paths.data_dir, args.nyu.base_path)
        self.train_path = os.path.join(self.nyu_base_path, self.args.nyu.train_path)
        self.eval_path = os.path.join(self.nyu_base_path, self.args.nyu.eval_path)

        # Load data filenames into memory, set per-split info (e.g. paths)
        if self.mode == 'online_eval':
            self.data_path = self.eval_path
            self.logger.debug(f"Filenames file: {self.args.nyu.filenames_file_eval}")
            with open(self.args.nyu.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            self.data_path = self.train_path
            self.logger.debug(f"Filenames file: {self.args.nyu.filenames_file_train}")
            with open(self.args.nyu.filenames_file_train, 'r') as f:
                self.filenames = f.readlines()


    def __len__(self):
        return len(self.filenames)   


    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])
        image_path = os.path.join(self.data_path, misc_utils.remove_leading_slash(sample_path.split()[0]))
        depth_path = os.path.join(self.data_path, misc_utils.remove_leading_slash(sample_path.split()[1]))

        has_valid_depth = False

        # Load files from disk
        image = Image.open(image_path)
        try:
            depth_gt = Image.open(depth_path)
            has_valid_depth = True
        except IOError:
            depth_gt = None
            self.logger.debug(f"Missing depth GT for {image_path}")

        # Training split
        if self.mode == 'train':
            assert has_valid_depth, f"Error: training example missing valid depth GT ({image_path}, {depth_path})."

        # Eval/validation split
        else:   
            pass            

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
        
            
