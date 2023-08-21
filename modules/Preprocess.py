# Preprocess.py
# Contains preprocessing steps to be performed on a dataset

import os, sys, logging
from types import NoneType
from typing import Dict

from PIL import Image
import numpy as np
import kornia
import torch
import torch.nn as nn
from torchvision import transforms

class Preprocess(nn.Module):
    """Class implementing basic preprocessing of PIL-based RGB and depth images.

    Responsible for conversion to torch.Tensors, normalisation, and cropping if needed.
    """
    def __init__(self, args: Dict, mode='train'):
        super().__init__()
        self.args = args
        self.mode = mode
        self.logger = logging.getLogger(f"{__name__}({self.mode})")

        self.logger.info(f"Creating Preprocessor")
        self.logger.debug(f"Using dataset: {args.basic.dataset}")

        self.imgsz = self.args[self.args.basic.dataset].dimensions_train
        self.random_crop = kornia.augmentation.RandomCrop(size=(self.imgsz[0], self.imgsz[1]))
        self.random_rotate_bilinear = kornia.augmentation.RandomRotation(degrees=self.args[self.args.basic.dataset].degree, resample='bilinear')
        self.random_rotate_nearest = kornia.augmentation.RandomRotation(degrees=self.args[self.args.basic.dataset].degree, resample='nearest')

        if self.args.basic.dataset == "nyu":
            self.image_norm_factor = self.args.nyu.image_norm_factor
            self.depth_norm_factor = self.args.nyu.depth_norm_factor
        elif self.args.basic.dataset == "kitti":
            self.image_norm_factor = self.args.kitti.image_norm_factor
            self.depth_norm_factor = self.args.kitti.depth_norm_factor
        else:
            self.logger.warning(f"Error: unrecognised args.basic.dataset: {args.basic.dataset}")
            sys.exit()


    def PIL_depth_to_metres_tensor(
        self,
        depth: Image.Image,
        factor: float | NoneType = None
        ) -> torch.Tensor:
        """Applies normalisation to PIL depth maps, converts units to metres,
        and returns as a torch.Tensor.
        
        Args:
            depth (PIL.Image.Image): input depth map
        
        Returns:
            depth (torch.Tensor, CxHxW): Normalised depth map as torch.Tensor
        """
        if factor is None:
            factor = self.depth_norm_factor
        depth = np.asarray(depth, dtype=np.float32)
        depth = np.expand_dims(depth, axis=2)
        depth = kornia.utils.image_to_tensor(depth, keepdim=True).float()
        depth /= factor  # Return result in metres.
        return depth


    def PIL_image_to_01_tensor(
        self,
        image: Image.Image,
        factor: float | NoneType = None
        ) -> torch.Tensor:
        """Convert PIL image to torch.Tensor in the range [0,1], as long as self.image_norm_factor
        has been set correctly in __init__().
        
        Args:
            image (PIL.Image.Image): input image
            factor (float, optional): optional normalisation factor override.
        
        Returns:
            image (torch.Tensor, CxHxW): Normalised image as torch.Tensor
        """
        if factor is None:
            factor = self.image_norm_factor
        image = np.asarray(image, dtype=np.float32)
        image = kornia.utils.image_to_tensor(image, keepdim=True).float()
        image /= factor
        return image


    def do_kb_crop(self, image: torch.Tensor, depth_gt: torch.Tensor | NoneType):
        """Applies KITTI benchmark crop to the list of image-like tensors provided.
        
        Args:
            image (torch.Tensor (3xHxW)): Input image
            depth (torch.Tensor (1xHxW)): Ground-truth depth map
        
        Returns:
            (image, depth) (torch.Tensor (3xHxW), torch.Tensor (1xHxW)): tuple of cropped image and depth
        """

        height = image.shape[1]
        width = image.shape[2]
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)

        image = image[:, top_margin:top_margin + 352, left_margin:left_margin + 1216]   # For CxHxW
        if depth_gt is not None and self.mode == 'online_eval':    # Sometimes images don't have corresponding depth.
            depth_gt = depth_gt[:, top_margin:top_margin + 352, left_margin:left_margin + 1216]

        return image, depth_gt


    @torch.no_grad()
    def forward(self, sample: Dict):
        """Convert to tensor, apply non-augmentation transforms (e.g. NYU standard cropping).

        Args:
            sample (Dict: Sample dict containing 'image' and 'depth_gt' keys (and others). 'depth_gt' key is optional
        
        Returns:
            sample (Dict: Sample dict containing 'image' and 'depth_gt' keys (and others). 'depth_gt' key is optional
        """
        assert 'image' in sample, "Sample contains no image."
        
        # Convert to torch.Tensor, compress image to range [0,1] and convert depth units to metres
        sample['image'] = self.PIL_image_to_01_tensor(sample['image'])
        if 'depth_gt' in sample and sample['depth_gt'] is not None:
            sample['depth_gt'] = self.PIL_depth_to_metres_tensor(sample['depth_gt'])

        if self.args[self.args.basic.dataset].do_kb_crop:
            sample['image'], sample['depth_gt'] = self.do_kb_crop(image=sample['image'], depth_gt=sample['depth_gt'])

        if self.args.basic.dataset == "nyu" and self.mode == "train":
            # See https://github.com/cleinc/bts/blob/master/pytorch/bts_dataloader.py#L117
            sample['image'] = transforms.functional.crop(sample['image'], top=45, left=43, height=427, width=565)
            sample['depth_gt'] = transforms.functional.crop(sample['depth_gt'], top=45, left=43, height=427, width=565)
        
        ## Random rotate and crop: used to be on GPU (in DataAugmentation.py) but moved because KITTI is variably-sized
        if self.mode == "train":
            # Unsqueeze and then re-squeeze at the end to play nice with the Kornia functions
            sample['image'] = sample['image'].unsqueeze(0)
            if 'depth_gt' in sample and sample['depth_gt'] is not None:
                sample['depth_gt'] = sample['depth_gt'].unsqueeze(0)

            # Random rotate (if requested)
            if self.args[self.args.basic.dataset].do_random_rotate is True:
                sample['image'] = self.random_rotate_bilinear(sample['image'])
                if 'depth_gt' in sample and sample['depth_gt'] is not None:
                    sample['depth_gt'] = self.random_rotate_nearest(sample['depth_gt'], params=self.random_rotate_bilinear._params)

            # Random crop (always)
            sample['image'] = self.random_crop(sample['image'])
            if 'depth_gt' in sample and sample['depth_gt'] is not None:
                sample['depth_gt'] = self.random_crop(sample['depth_gt'], params=self.random_crop._params)
                sample['depth_gt'] = sample['depth_gt'].squeeze(0)
            sample['image'] = sample['image'].squeeze(0)

        return sample

