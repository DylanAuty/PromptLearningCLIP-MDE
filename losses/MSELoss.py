# MSELoss.py
# Wrapper around nn.MSELoss to allow passing kwargs

import torch.nn as nn

class MSELoss(nn.Module):
    """Wrapper around nn.MSELoss to allow the passing of kwargs."""
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    
    def forward(self, depth_pred, depth_gt, **kwargs):
        return self.mse_loss(input=depth_pred, target=depth_gt)