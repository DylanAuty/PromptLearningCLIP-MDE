import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl

class LearnableTokenEmbeddings(pl.LightningModule):
    """Class to implement a set of learnable tokens for use as part of learned prompts. The mode argument determines 
    how the token is learned. These tokens are to be used within clip as inputs to the transformer, NOT as inputs
    to the embedding module prior to the transformer.

    :param count (int, default 1): How many learnable tokens to implement
    :param mode (string): How the token is learned.
    :param dim (int, default 512): the required output dimensions of each token.

    """

    def __init__(self, count=1, mode="static", dim=512):
        super().__init__()

        self.mode = mode

        match self.mode:
            case "static":
                # A set of nn.Parameters of the requested dimensions.
                self.embeddings = nn.Embedding(count, dim)
                nn.init.normal_(self.embeddings.weight, std=0.02)   # Init settings following CLIP's clip.initialize_parameters() fn
            case _:
                sys.exit("Error: LearnableTokenEmbeddings mode not recognised.")

    
    def forward(self, input):
        match self.mode:
            case "static":
                return self.embeddings(torch.zeros(1, dtype=torch.int, device=self.device))
            case _:
                sys.exit("Error: Other modes not implemented yet.")

    
    def __getitem__(self, idx):
        sys.exit("Error: [] indexing is deprecated for this module.")
        match self.mode:
            case "static":
                return self.embeddings(torch.IntTensor([idx]))
            case _:
                sys.exit("Error: LearnableTokenEmbeddings __getitem__ may not work with given mode (not implemented yet?)")

            