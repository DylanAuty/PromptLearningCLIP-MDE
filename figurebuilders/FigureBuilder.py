# FigureBuilder.py
from typing import Tuple
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import colorcet as cc
import numpy as np
import torch

class FigureBuilder():
    """A class for setting up a grid of images relating to depth experiments.

    Purpose is to centralise all code for handling ranges, colour schemes, etc, 
    so that consistent figures are generated.
    Use by calling add_image() with the batched batch['image'], depths, etc., and then
    access the figure and do whatever with it.
    When finished, clear the figure by calling the reset() method.
    """

    def __init__(self, args, num_samples, extra_rgb=0, extra_titles=None):
        self.args = args
        self.num_samples = num_samples
        self.extra_titles = extra_titles

        # Basic formatting for the matplotlib figure
        self.num_columns = 3    # Image, Depth, Depth Prediction.
        self.num_columns += extra_rgb

        # Setting aspect ratio according to dataset EVAL dimensions
        dataset_val_dims = self.args[self.args.basic.dataset].dimensions_test
        if self.args[self.args.basic.dataset].do_kb_crop:
            dataset_val_dims = [352, 1216]
        aspect_ratio = dataset_val_dims[0] / dataset_val_dims[1]

        plt.axis('off')
        width = self.num_columns * (7/3)
        height = self.num_samples * width / self.num_columns * aspect_ratio + 0.3
        self.fig, self.ax = plt.subplots(self.num_samples, self.num_columns, figsize=(width, height))
        self.setup_axes()

        # Get colour map to use
        self.cmap1 = matplotlib.cm.get_cmap("inferno_r")
        self.cmap1.set_bad(color="white")
        self.cmap1.set_under(color="white")
        self.cmap2 = matplotlib.cm.get_cmap("inferno_r")

    
    def setup_axes(self):
        """Sets up the axes of the plot.
        """
        [ax.clear() for ax in self.ax.ravel()]
        self.curr_sample = 0
        self.ax[self.curr_sample, 0].set_title('RGB')
        self.ax[self.curr_sample, 1].set_title('G.T. Depth')
        self.ax[self.curr_sample, 2].set_title('Pred. Depth')

        if self.num_columns > 3:
            for i in range(3, self.num_columns):
                self.ax[self.curr_sample, i].set_title(self.extra_titles[i - 3])

        [axi.set_axis_off() for axi in self.ax.ravel()]


    def build(self, batch):
        """Method to be called to set up a plot of self.num_samples samples, or fewer.

        All samples will be from the same batch, and all inputs are batched (BxCxHxW).
        Once this has set the figure up, it can be written to tensorboard or similar.

        Args:
            batch (Dict | Tuple): a batch containing keys 'image', 'depth_gt', 'depth_pred'.

        Returns:
            fig (matplotlib figure): A reference to the figure, for easy plotting/saving to tensorboard
        """

        # Add as many samples as possible from the batch, up to the max number
        # if there's fewer samples in the batch, then do fewer
        if self.num_samples < batch['image'].shape[0]:
            samples_to_plot = self.num_samples
        else:
            samples_to_plot = batch['image'].shape[0]

        for self.curr_sample in range(0, samples_to_plot):
            # The image can be safely added to the plot.
            # batch['image']
            rgb_min = torch.min(batch['image'][self.curr_sample, :])
            rgb_max = torch.max(batch['image'][self.curr_sample, :])
            batch['image'][self.curr_sample, :] = (batch['image'][self.curr_sample, :] - rgb_min) / (rgb_max - rgb_min)
            self.ax[self.curr_sample, 0].imshow(np.transpose(batch['image'][self.curr_sample, :].detach().cpu().numpy(), (1, 2, 0)))
            self.ax[self.curr_sample, 0].axis('off')
            # GT Depth
            # vmin = torch.min(batch['depth_gt'][self.curr_sample, 0])
            vmin = self.args[self.args.basic.dataset].min_depth
            vmax = torch.max(batch['depth_gt'][self.curr_sample, 0])
            self.ax[self.curr_sample, 1].imshow(batch['depth_gt'][self.curr_sample, 0].detach().cpu().numpy(), vmin=vmin, vmax=vmax, cmap=self.cmap1)
            self.ax[self.curr_sample, 1].axis('off')

            # Predicted Depth
            # Uses vmin and vmax from batch['depth_gt'] so they're comparable
            self.ax[self.curr_sample, 2].imshow(batch['depth_pred'][self.curr_sample, 0].detach().cpu().numpy(), vmin=vmin, vmax=vmax, cmap=self.cmap2)
            self.ax[self.curr_sample, 2].axis('off')

        # Re-set the aspect ratio based on the plotted images
        aspect_ratio = batch['image'][0].shape[1] / batch['image'][0].shape[2]
        width = self.num_columns * (7/3)
        height = self.num_samples * width / self.num_columns * aspect_ratio + 0.3
        self.fig.set_figheight(height)
        self.fig.set_figwidth(width)

        # Plot is populated now, so fiddle with the layout a bit and set up the DPI
        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.02, wspace=0.04)
        self.fig.dpi = 250

        return self.fig


    def reset(self):
        """A convenience function. After setting up with self.add_image() and plotting with
        tensorboard or similar, self.reset() should be called in order to clear the figure and reinitialise the
        counter self.curr_sample. If this isn't reset, it will refuse to plot (to avoid filling the log with
        pointless figures).
        """
        self.setup_axes()
