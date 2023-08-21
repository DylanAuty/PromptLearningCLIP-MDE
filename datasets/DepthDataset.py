# DepthDataset.py
# A wrapper class for the different datasets that will be used
# Useful because certain augmentations are shared between datasets.

import os, sys, logging
from types import NoneType

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import misc_utils

class DepthDataset(Dataset):
    """A wrapper class for various image/depth datasets.
    
    Responsible for dataset-specific file loading from disk. Not responsible for preprocessing
    (preprocessing module is passed as the transform argument, and is applied at the end of the
    __getitem__() method.)
    
    Intended to be overwritten for specific datasets, but contains some things common
    to each of them. Datasets are expected to return batches as dicts, with at least the keys
    'image' and 'depth' for the RGB image and ground-truth depth respectively.

    The image returned must lie in the range [0, 1]. Normalisation according to ImageNet statistics
    is performed elsewhere, not here. Augmentation is also not performed here.

    """
    def __init__(self, args):
        super().__init__()
        # args = self.args

