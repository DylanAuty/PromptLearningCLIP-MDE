# LossWrapper.py
# Implements a generic loss function that configures itself based on the
# args passed to it.

import os, sys, logging
import torch
import torch.nn as nn

from losses.MSELoss import MSELoss
from losses.SILogLoss import SILogLoss
from losses.BinsChamferLoss import BinsChamferLoss


class LossWrapper(nn.Module):
    """Implements a generic loss function that configures itself based on the
    args passed to it.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.logger = logging.getLogger(__name__)

        possible_losses = [
            'mse',
            'silog',
            'bins_chamfer'
        ]

        assert 'loss' in self.args, f"Error: loss section missing from args file/dict."
        assert 'names' in self.args.loss, f"Error: Loss names key not found in args.loss."
        assert len(self.args.loss.names) >= 1, "Error: no loss names given"
        assert all([name in possible_losses for name in self.args.loss.names]), f"Error: unrecognised loss function"
        assert 'coeffs' in self.args.loss, f"Error: Loss component coefficients array not found."
        assert len(self.args.loss.coeffs) == len(self.args.loss.names), "Error: mismatched number of loss components and coefficients"

        self.loss_components = nn.ModuleList()

        for i, name in enumerate(self.args.loss.names):
            match name:
                case 'mse':
                    self.logger.info("Using MSE loss")
                    self.loss_components.append(MSELoss())
                case 'silog':
                    self.logger.info("Using SILog loss")
                    self.loss_components.append(SILogLoss(self.args))
                case 'bins_chamfer':
                    self.logger.info("Using Bins Chamfer loss")
                    self.loss_components.append(BinsChamferLoss(self.args))

        
    def forward(self, depth_pred, depth_gt, depth_mask, output_named_tuple):
        """For genericity, accepts some essential arguments and a namedtuple of everything else that another
        loss function might need.

        Arguments should be named consistently between loss functions to make this work:
            image (torch.Tensor, Bx3xHxW):          If used, the RGB image
            depth_gt (torch.Tensor, Bx1xHxW):       Ground truth depth map
            depth_pred (torch.Tensor, Bx1xHxW):     Predicted depth map
            depth_mask (torch.BoolTensor, Bx1xHxW): If used, a mask of valid (True) and invalid (False) depth values.
            bin_edges (List):                       For binning losses. List of bin edges.
        """

        loss = 0
        for i, component in enumerate(self.loss_components):
            loss += self.args.loss.coeffs[i] * component(depth_pred, depth_gt, depth_mask, output_named_tuple)

        return loss
