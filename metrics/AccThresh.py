# AccThresh.py
# TorchMetrics implementation of accuracy-within-threshold metric
# I.e. proportion of points that are within the given error bound of being correct

from types import NoneType
import torch
import torch.nn as nn
import torchmetrics


class AccThreshRunningAvg(torchmetrics.Metric):
    """Implements running average of metric of proportion of points accurate within the given threshold.
    Often in the depth literature as delta 1.25^n, where n is 1, 2 or 3.
    i.e. mean(max((gt / pred), (pred / gt)) < threshold)
    """
    higher_is_better = True
    full_state_update = True


    def __init__(self, args, threshold: float):
        super().__init__()
        self.args = args
        self.threshold = threshold

        self.add_state("batch_count", default=torch.Tensor([0]), dist_reduce_fx="sum")
        self.add_state("running_avg", default=torch.Tensor([0]), dist_reduce_fx="mean")

    
    def update(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor):
        """Expects pre-masked depth_pred and depth_gt (i.e. only valid values.)"""
        thresh = torch.maximum((depth_gt / depth_pred), (depth_pred / depth_gt))
        val = torch.mean((thresh < self.threshold).type(torch.float))
        self.running_avg = (val + self.running_avg * self.batch_count) / (self.batch_count + 1)
        self.batch_count += 1
    

    def compute(self):
        return self.running_avg


class AccThresh(torchmetrics.Metric):
    """Implements metric of proportion of points accurate within the given threshold.
    Often in the depth literature as delta 1.25^n, where n is 1, 2 or 3.
    i.e. mean(max((gt / pred), (pred / gt)) < threshold)
    """
    higher_is_better = True
    full_state_update = False


    def __init__(self, args, threshold: float):
        super().__init__()
        self.args = args
        self.threshold = threshold

        self.add_state("thresh_metric", default=torch.Tensor([0]), dist_reduce_fx="sum")
        self.add_state("valid_pixel_count", default=torch.Tensor([0]), dist_reduce_fx="sum")

    
    def update(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor):
        """Expects pre-masked depth_pred and depth_gt (i.e. only valid values.)"""
        thresh = torch.maximum((depth_gt / depth_pred), (depth_pred / depth_gt))
        self.thresh_metric += torch.sum(thresh < self.threshold)
        self.valid_pixel_count += torch.numel(depth_gt)
    

    def compute(self):
        return (self.thresh_metric / self.valid_pixel_count).float()