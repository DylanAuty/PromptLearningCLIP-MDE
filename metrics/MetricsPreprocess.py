# MetricsPreprocess.py
# Apply clamping and de-nan/inf-ing of a depth map prior to computing metrics.

import torch
import torch.nn as nn


class MetricsPreprocess(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args


    def forward(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor, do_clamp=True):
        """Handles some postprocessing needed for metrics (esp. Garg and Eigen crops)
        """
        depth_pred = nn.functional.interpolate(depth_pred, depth_gt.shape[-2:], mode='bilinear', align_corners=True)

        # This follows BTS and Adabins implementations, which use np.isinf, which returns True for both +inf and -inf.
        depth_pred = depth_pred.nan_to_num(
            nan=self.args[self.args.basic.dataset].min_depth,
            posinf=self.args[self.args.basic.dataset].max_depth,
            neginf=self.args[self.args.basic.dataset].max_depth
        )

        depth_mask = (depth_gt > self.args[self.args.basic.dataset].min_depth) & (depth_gt <= self.args[self.args.basic.dataset].max_depth)

        if self.args[self.args.basic.dataset].garg_crop or self.args[self.args.basic.dataset].eigen_crop:
            gt_height, gt_width = depth_gt.shape[2], depth_gt.shape[3]
            eval_mask = torch.zeros((gt_height, gt_width), dtype=torch.bool, device=depth_gt.device)
            
            if self.args[self.args.basic.dataset].garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = True

            elif self.args[self.args.basic.dataset].eigen_crop:
                if self.args.basic.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = True
                else:
                    eval_mask[45:471, 41:601] = True
        
            depth_mask &= eval_mask

        return depth_pred, depth_mask

       