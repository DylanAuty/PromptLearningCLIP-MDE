# SILogLoss.py
# Contains an implementation of the Scale Invariant log loss used in the AdaBins paper.

import os, sys, logging
from types import NoneType
from typing import Union
import torch
import torch.nn as nn

class SILogLoss(nn.Module):
    """Scale-invariant log-loss as defined in Bhat et al.'s Adabins paper.
    
    Note that this implements the version in section 3.4 of that paper:
    SIloss = alpha * sqrt(mean(g^2) - (lambda/(T^2))(sum(g_i)^2))

    It does not use the implementation in the official code release for that paper.
     """
    def __init__(self, args):
        super().__init__()
        self.name = 'SILog'
        self.args = args

        # Loss hyperparams (alpha and lambda) following those of the AdaBins paper.
        self.alph = 10
        self.lam = 0.85


    def forward(self,
            depth_pred: torch.Tensor,
            depth_gt: torch.Tensor,
            depth_mask: torch.Tensor | NoneType=None,
            interpolate=True,
            **kwargs
        ):
        """
        Args:
            input (torch.Tensor, Bx1xHxW): The predicted depth map
            target (torch.Tensor, Bx1xHxW): The ground-truth depth map.
            depth_mask (torch.BoolTensor, Bx1xHxW, optional): Optional depth_mask of valid depth values
            interpolate (Boolean, default True): Whether to interpolate the input to match the target.
            **kwargs (dict): Any other kwargs passed from the loss wrapper. These get ignored.
        """
        if interpolate:
            depth_pred = nn.functional.interpolate(depth_pred, depth_gt.shape[-2:], mode='bilinear', align_corners=True)

        if depth_mask is not None:
            depth_pred = depth_pred[depth_mask]
            depth_gt = depth_gt[depth_mask]

        n_points = torch.numel(depth_pred)
        g = torch.log(depth_pred) - torch.log(depth_gt)

        # Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)    # Equivalent implementation from Adabins code
        Dg = (torch.sum(g ** 2) / n_points) - ((self.lam / (n_points ** 2)) * (torch.sum(g) ** 2))

        return self.alph * torch.sqrt(Dg)