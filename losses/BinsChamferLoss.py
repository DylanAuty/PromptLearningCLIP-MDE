# BinsChamferLoss.py
# Implements the AdaBins chamfer loss, as used in the AdaBins paper.
# Implementation adapted from the official implementation of AdaBins at:
# https://github.com/shariqfarooq123/AdaBins

import os, sys, logging
from types import NoneType
from typing import List, Union
import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance


class BinsChamferLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = 'ChamferLoss'
        self.args = args
        

    def forward(self,
            depth_pred, depth_gt, depth_mask, output_named_tuple
        ):
        bin_edges = output_named_tuple.bin_edges
        # bin_edges = output_named_tuple[1]
        bin_centers = 0.5 * (bin_edges[:, 1:] + bin_edges[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1

        target_points = depth_gt.flatten(1)  # n, hwc
        depth_mask = depth_mask.flatten(1)
        target_points = [p[m] for p, m in zip(target_points, depth_mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(depth_gt.device)
        target_points = nn.utils.rnn.pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss
