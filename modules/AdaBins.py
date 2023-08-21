# AdaBins.py
# Implementation of the AdaBins model.
# Code adapted from the official implementation of AdaBins at https://github.com/shariqfarooq123/AdaBins

import os, sys, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import namedtuple

from modules.miniViT import mViT
from modules.DenseFeatureExtractor import DenseFeatureExtractor

class AdaBins(nn.Module):
    """Implements the AdaBins model. Implementation is adapted from the official
    implementation, which is found at https://github.com/shariqfarooq123/AdaBins.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.logger = logging.getLogger(__name__)

        self.n_bins = self.args.adabins.n_bins
        self.num_decoded_channels = 128		# By default this is 128 but will change with different experiments.

        # These two lists get used if args.optimizer.slow_encoder is set (to put a lower learning rate on the encoder.)
        self._encoder_params_module_list = []
        self._non_encoder_params_module_list = []
        self._frozen_params_module_list = []

        self.ReturnType = namedtuple('ReturnType', ['depth_pred', 'bin_edges'])

        self.logger.info("Building model")

        # Encoder/decoder with skip connections
        self.dense_feature_extractor = DenseFeatureExtractor(self.args)
        self._encoder_params_module_list.append(self.dense_feature_extractor.encoder)
        self._non_encoder_params_module_list.append(self.dense_feature_extractor.decoder)

        # Adaptive binning layer (miniViT)
        max_seq_len = 1200 if self.args[self.args.model.name].get('do_final_upscale') else 500
        self.adaptive_bins_layer = mViT(self.num_decoded_channels, n_query_channels=128, patch_size=16,
                                        dim_out=self.n_bins,
                                        embedding_dim=128, norm='linear',
                                        max_seq_len=max_seq_len)
        self._non_encoder_params_module_list.append(self.adaptive_bins_layer)

        # Output block
        self.conv_out = nn.Sequential(
                            nn.Conv2d(128, self.n_bins, kernel_size=1, stride=1, padding=0),
                            nn.Softmax(dim=1)
                        )
        self._non_encoder_params_module_list.append(self.conv_out)


    def get_encoder_params(self):
        for m in self._encoder_params_module_list:
            yield from m.parameters()


    def get_non_encoder_params(self):
        for m in self._non_encoder_params_module_list:
            yield from m.parameters()


    def get_frozen_params(self):
        for m in self._frozen_params_module_list:
            yield from m.parameters()


    def forward(self, image):
        unet_out = self.dense_feature_extractor(image)

        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)	# Gives bin logits
        
        bin_widths = (self.args[self.args.basic.dataset].max_depth - self.args[self.args.basic.dataset].min_depth) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.args[self.args.basic.dataset].min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        depth_pred = torch.sum(out * centers, dim=1, keepdim=True)

        return self.ReturnType(depth_pred=depth_pred, bin_edges=bin_edges)
