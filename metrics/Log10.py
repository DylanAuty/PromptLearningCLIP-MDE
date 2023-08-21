# Log10.py
# TorchMetrics implementation of Log10 (mean abs difference of base 10 logs)

from types import NoneType
import torch
import torch.nn as nn
import torchmetrics


class Log10RunningAvg(torchmetrics.Metric):
    """Implements Log 10 metric (mean of abs of difference of base 10 logs), as a running average.
    """
    higher_is_better = True
    full_state_update = True


    def __init__(self, args):
        super().__init__()
        self.args = args

        self.add_state("batch_count", default=torch.Tensor([0]), dist_reduce_fx="sum")
        self.add_state("running_avg", default=torch.Tensor([0]), dist_reduce_fx="mean")

    
    def update(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor):
        """Expects pre-masked depth_pred and depth_gt (i.e. only valid values.)"""
        val = torch.mean(torch.abs(torch.log10(depth_gt) - torch.log10(depth_pred)))

        self.running_avg = (val + self.running_avg * self.batch_count) / (self.batch_count + 1)
        self.batch_count += 1
    

    def compute(self):
        return self.running_avg


class Log10(torchmetrics.Metric):
    """Implements Log 10 metric (mean of abs of difference of base 10 logs).
    """
    higher_is_better = True
    full_state_update = False


    def __init__(self, args):
        super().__init__()
        self.args = args

        self.add_state("total_abs_diffs_of_log10s", default=torch.Tensor([0]), dist_reduce_fx="sum")
        self.add_state("valid_pixel_count", default=torch.Tensor([0]), dist_reduce_fx="sum")

    
    def update(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor):
        """Expects pre-masked depth_pred and depth_gt (i.e. only valid values.)"""
        thresh = torch.maximum((depth_gt / depth_pred), (depth_pred / depth_gt))

        self.total_abs_diffs_of_log10s += torch.sum(torch.abs(torch.log10(depth_gt) - torch.log10(depth_pred)))
        self.valid_pixel_count += torch.numel(depth_gt)
    

    def compute(self):
        return (self.total_abs_diffs_of_log10s / self.valid_pixel_count).float()