# MainRunnerLM.py
# Contains definition of the pytorch LightningModule used in this work.
# Actual model definitions (nn.Module) are in their own files.

import os, sys, logging
import torch
import torchvision
import numpy as np
import pandas as pd

import torch.nn as nn
import pytorch_lightning as pl

from datasets.NYUD2 import NYUD2
from datasets.KITTI import KITTI
from datasets.dataloader import DepthDataLoader # Original Adabins dataloader
from modules.Preprocess import Preprocess
from modules.DataAugmentation import DataAugmentation
from modules.AdaBins import AdaBins
from modules.DepthCLIP import DepthCLIP

from losses.LossWrapper import LossWrapper
from metrics.MetricsPreprocess import MetricsPreprocess
from metrics.AbsRel import AbsRel, AbsRelRunningAvg
from metrics.SqRel import SqRel, SqRelRunningAvg
from metrics.RMSE import RMSE, RMSERunningAvg
from metrics.RMSELog import RMSELog, RMSELogRunningAvg
from metrics.AccThresh import AccThresh, AccThreshRunningAvg
from metrics.Log10 import Log10, Log10RunningAvg

from figurebuilders.FigureBuilder import FigureBuilder

import matplotlib
from matplotlib import pyplot as plt

class MainRunnerLM(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        # Data preprocessing/augmentation/normalization modules
        if self.args.basic.dataset in ["kitti", "nyu"] and not self.args.basic.get("use_adabins_dataloader"):
            self.train_preprocessor = Preprocess(self.args, mode='train')
            self.val_preprocessor = Preprocess(self.args, mode='online_eval')
            self.data_augmentation = DataAugmentation(self.args)
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    # ImageNet statistics
        self.unnormalize = torchvision.transforms.Compose([
                                torchvision.transforms.Normalize(mean = [ 0.0, 0.0, 0.0 ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1.0, 1.0, 1.0 ]),
                               ])
        
        # Metrics and metrics preprocessing
        self.metrics_preprocess = MetricsPreprocess(self.args)

        # For making neat grids of batches + predictions
        match self.args.model.name:
            case "adabins" | "depthclip":
                self.figure_builder = FigureBuilder(self.args, num_samples=min(self.args.basic.batch_size, 4)) # Up to 4 samples from the batch
            case _:
                sys.exit("Error: not implemented")

        # Example input for use when building model graph
        if self.args[self.args.basic.dataset].do_kb_crop:
            # 376, 1241
            eg_height = 352
            eg_width = 1216
        else:
            eg_height = self.args[self.args.basic.dataset].dimensions_train[0]
            eg_width = self.args[self.args.basic.dataset].dimensions_train[1]
        # self.example_input_array = {
        #     'image': torch.rand((self.args.basic.batch_size, 3, eg_height, eg_width)),
        #     'depth': torch.rand((self.args.basic.batch_size, 1, eg_height, eg_width)),
        # }
        self.example_input_array = (
            torch.rand((self.args.basic.batch_size, 3, eg_height, eg_width)),
            torch.rand((self.args.basic.batch_size, 1, eg_height, eg_width)),
        )

        self.var_acc = 0.0
        self.var_acc_counter = 0

        self.most_recent_train_batch = None   # To be used to store the most recent batch.
        self.most_recent_val_batch = None   # To be used to store the most recent batch.

        # Model and loss definitions
        match self.args.model.name:
            case "adabins":
                self.model = AdaBins(self.args)
            case "depthclip":
                self.model = DepthCLIP(self.args)
            case _:
                sys.exit(f"Error: unrecognised model ({self.args.model.get('name')})")

        # For prediction/inference mode: a dict for storing per-example metrics, paths, filenames.
        self.prediction_dict = {}

        self.loss = LossWrapper(self.args)

        # Metric definitions
        self.abs_rel = AbsRel(self.args)
        self.sq_rel = SqRel(self.args)
        self.rmse = RMSE(self.args)
        self.rmse_log = RMSELog(self.args)
        self.log10 = Log10(self.args)
        self.acc_1 = AccThresh(self.args, threshold=1.25)
        self.acc_2 = AccThresh(self.args, threshold=1.25 ** 2)
        self.acc_3 = AccThresh(self.args, threshold=1.25 ** 3)

        self.abs_rel_ra = AbsRelRunningAvg(self.args)
        self.sq_rel_ra = SqRelRunningAvg(self.args)
        self.rmse_ra = RMSERunningAvg(self.args)
        self.rmse_log_ra = RMSELogRunningAvg(self.args)
        self.log10_ra = Log10RunningAvg(self.args)
        self.acc_1_ra = AccThreshRunningAvg(self.args, threshold=1.25)
        self.acc_2_ra = AccThreshRunningAvg(self.args, threshold=1.25 ** 2)
        self.acc_3_ra = AccThreshRunningAvg(self.args, threshold=1.25 ** 3)


    def forward(self, *batch, **kwargs):
        """Forward method.
        
        Args:
            batch (tuple of image, depth): the input image and corresponding ground truth.
        """
        image, depth_gt = batch
        # image, depth_gt = batch["image"], batch["depth"]

        return self.model(image)


    def training_step(self, batch, batch_idx):
        image, depth_gt = batch["image"], batch["depth"]

        output_named_tuple = self.model(image)
        depth_pred = output_named_tuple.depth_pred  # No clamping is used during training, but min/max clamping and nan/inf removal are used during eval/val
        
        depth_mask = (depth_gt > self.args[self.args.basic.dataset].min_depth) # BTS and AdaBins only use a minimum mask during training, but use both during validation/evaluation

        loss = self.loss(depth_pred=depth_pred, depth_gt=depth_gt, depth_mask=depth_mask, output_named_tuple=output_named_tuple)

        self.most_recent_train_batch = {
            'image': image,
            'depth_gt': depth_gt,
            'depth_pred': nn.functional.interpolate(depth_pred, depth_gt.shape[-2:], mode='bilinear', align_corners=True),
            'loss': loss
        }
        self.log("train/loss", loss, on_step=True)

        return loss


    def training_epoch_end(self, training_step_outputs):
        self.logger.experiment.add_figure(tag="train/samples", figure=self.figure_builder.build(self.most_recent_train_batch), global_step=self.global_step)
        self.figure_builder.reset()        
    

    def validation_step(self, batch, batch_idx):
        image, depth_gt = batch["image"], batch["depth"]

        # Validation run on image and its mirror, then averaged, following previous work.
        # Regular predictions:
        output_named_tuple = self.model(image)
        depth_pred = output_named_tuple.depth_pred
        depth_pred = torch.clamp(
            depth_pred,
            min=self.args[self.args.basic.dataset].min_depth,
            max=self.args[self.args.basic.dataset].max_depth
        )
        # nan and inf handling done in metrics_preprocess

        logits = output_named_tuple.depth_logits
        logit_var = logits.var(dim=1).reshape(logits.shape[0], -1)
        self.var_acc += logit_var.sum()
        self.var_acc_counter += logit_var.shape[-1]
        
        # Mirrored predictions:
        output_named_tuple_mirror = self.model(image.flip(dims=[3]))
        depth_pred_mirror = output_named_tuple_mirror.depth_pred
        depth_pred_mirror = depth_pred_mirror.flip(dims=[3])
        depth_pred_mirror = torch.clamp(
            depth_pred_mirror,
            min=self.args[self.args.basic.dataset].min_depth,
            max=self.args[self.args.basic.dataset].max_depth
        )
        # nan and inf handling done in metrics_preprocess

        depth_pred_final = 0.5 * (depth_pred + depth_pred_mirror)

        depth_mask = (depth_gt > self.args[self.args.basic.dataset].min_depth) & (depth_gt <= self.args[self.args.basic.dataset].max_depth)

        loss = self.loss(depth_pred=depth_pred_final, depth_gt=depth_gt, depth_mask=depth_mask, output_named_tuple=output_named_tuple)

        self.most_recent_val_batch = {
            'image': image,
            'depth_gt': depth_gt.clone(),
            'depth_pred': nn.functional.interpolate(depth_pred_final.clone(), depth_gt.shape[-2:], mode='bilinear', align_corners=True),
            'loss': loss
        }

        # Apply any crops (eigen/garg) if necessary (metrics preprocessing), then apply validity mask.
        depth_pred_m, depth_mask_m = self.metrics_preprocess(depth_pred=depth_pred_final.clone(), depth_gt=depth_gt.clone())
        depth_pred_m, depth_gt_m = depth_pred_m[depth_mask_m], depth_gt[depth_mask_m]
        
        # Compute metrics
        self.abs_rel(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.sq_rel(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.rmse(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.rmse_log(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.log10(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_1(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_2(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_3(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())

        self.abs_rel_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.sq_rel_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.rmse_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.rmse_log_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.log10_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_1_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_2_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_3_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())

        # Log metrics and loss
        self.log("metrics/abs_rel", self.abs_rel, on_epoch=True)
        self.log("metrics/sq_rel", self.sq_rel, on_epoch=True)
        self.log("metrics/rmse", self.rmse, on_epoch=True)
        self.log("metrics/rmse_log", self.rmse_log, on_epoch=True)
        self.log("metrics/log10", self.log10, on_epoch=True)
        self.log("metrics/acc_1", self.acc_1, on_epoch=True)
        self.log("metrics/acc_2", self.acc_2, on_epoch=True)
        self.log("metrics/acc_3", self.acc_3, on_epoch=True)

        self.log("metrics_ra/abs_rel_ra", self.abs_rel_ra, on_epoch=True)
        self.log("metrics_ra/sq_rel_ra", self.sq_rel_ra, on_epoch=True)
        self.log("metrics_ra/rmse_ra", self.rmse_ra, on_epoch=True)
        self.log("metrics_ra/rmse_log_ra", self.rmse_log_ra, on_epoch=True)
        self.log("metrics_ra/log10_ra", self.log10_ra, on_epoch=True)
        self.log("metrics_ra/acc_1_ra", self.acc_1_ra, on_epoch=True)
        self.log("metrics_ra/acc_2_ra", self.acc_2_ra, on_epoch=True)
        self.log("metrics_ra/acc_3_ra", self.acc_3_ra, on_epoch=True)

        self.log("val/loss", loss, on_epoch=True, sync_dist=True)

        return loss


    def validation_epoch_end(self, validation_step_outputs):
        # This dict is used at the end of training to get a plaintext, easily copy-pastable version of the most recent metrics.
        self.last_metrics_dict = {
            "abs_rel": self.abs_rel.compute(),
            "sq_rel": self.sq_rel.compute(),
            "rmse": self.rmse.compute(),
            "rmse_log": self.rmse_log.compute(),
            "log10": self.log10.compute(),
            "acc_1": self.acc_1.compute(),
            "acc_2": self.acc_2.compute(),
            "acc_3": self.acc_3.compute(),
            "abs_rel_ra": self.abs_rel_ra.compute(),
            "sq_rel_ra": self.sq_rel_ra.compute(),
            "rmse_ra": self.rmse_ra.compute(),
            "rmse_log_ra": self.rmse_log_ra.compute(),
            "log10_ra": self.log10_ra.compute(),
            "acc_1_ra": self.acc_1_ra.compute(),
            "acc_2_ra": self.acc_2_ra.compute(),
            "acc_3_ra": self.acc_3_ra.compute(),
        }
        self.logger.experiment.add_figure(tag="val/samples", figure=self.figure_builder.build(self.most_recent_val_batch), global_step=self.global_step)
        self.figure_builder.reset()

        print("LOGIT VARIANCE: ")
        print(self.var_acc / self.var_acc_counter)

    
    def on_train_end(self):
        log_str = f"abs_rel, sq_rel, rms, rmsl, log10, d1, d2, d3:  \n {self.last_metrics_dict['abs_rel']}, {self.last_metrics_dict['sq_rel']}, {self.last_metrics_dict['rmse']}, {self.last_metrics_dict['rmse_log']}, {self.last_metrics_dict['log10']}, {self.last_metrics_dict['acc_1']}, {self.last_metrics_dict['acc_2']}, {self.last_metrics_dict['acc_3']}  \n ==#==  \nabs_rel_ra, sq_rel_ra, rms_ra, rmsl_ra, log10_ra, d1_ra, d2_ra, d3_ra:  \n{self.last_metrics_dict['abs_rel_ra']}, {self.last_metrics_dict['sq_rel_ra']}, {self.last_metrics_dict['rmse_ra']}, {self.last_metrics_dict['rmse_log_ra']}, {self.last_metrics_dict['log10_ra']}, {self.last_metrics_dict['acc_1_ra']}, {self.last_metrics_dict['acc_2_ra']}, {self.last_metrics_dict['acc_3_ra']}"
        self.logger.experiment.add_text("metrics/all", log_str, global_step=self.global_step)
        hparam_dict = {
            "batch size": self.args.basic.batch_size,
            "use_swa": ("use_swa" in self.args.optimizer and self.args.optimizer.use_swa),
            "model": self.args.model.name,
            "encoder name": self.args[self.args.model.name].get("encoder_name"),
            "clip": self.args[self.args.model.name].get("clip"),
            "current epoch": self.current_epoch,
            "precision": self.precision,
        }
        
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """ Prediction step will take a prediction (without test-time augmentation), run evaluation on it as well, and
        save the whole batch (input and outputs, including any auxiliary outputs) plus the metrics to disk. 
        Whatever gets returned here ends up in a list (handled by lightning), and that list gets returned when calling
        trainer.predict() with a dataloader.
        """

        image, depth_gt = batch["image"], batch["depth"]

        # Regular predictions:
        output_named_tuple = self.model(image)
        depth_pred = output_named_tuple.depth_pred
        depth_pred = torch.clamp(
            depth_pred,
            min=self.args[self.args.basic.dataset].min_depth,
            max=self.args[self.args.basic.dataset].max_depth
        )
        # nan and inf handling done in metrics_preprocess

        depth_mask = (depth_gt > self.args[self.args.basic.dataset].min_depth) & (depth_gt <= self.args[self.args.basic.dataset].max_depth)

        loss = self.loss(depth_pred=depth_pred, depth_gt=depth_gt, depth_mask=depth_mask, output_named_tuple=output_named_tuple)

        self.most_recent_pred_batch = {
            'image': image,
            'depth_gt': depth_gt.clone(),
            'depth_pred': nn.functional.interpolate(depth_pred.clone(), depth_gt.shape[-2:], mode='bilinear', align_corners=True),
            'loss': loss
        }

        depth_pred = nn.functional.interpolate(depth_pred.clone(), depth_gt.shape[-2:], mode='bilinear', align_corners=True)


        # Apply any crops (eigen/garg) if necessary (metrics preprocessing), then apply validity mask.
        depth_pred_m, depth_mask_m = self.metrics_preprocess(depth_pred=depth_pred.clone(), depth_gt=depth_gt.clone())
        depth_pred_m, depth_gt_m = depth_pred_m[depth_mask_m], depth_gt[depth_mask_m]
        
        # Compute metrics
        self.abs_rel(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.sq_rel(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.rmse(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.rmse_log(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.log10(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_1(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_2(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_3(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())

        self.abs_rel_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.sq_rel_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.rmse_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.rmse_log_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.log10_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_1_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_2_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_3_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())

        # Turn depth maps into RGB
        match self.args.get("inf_save_mode"):
            case "all" | "some":
                if self.args.get("inf_save_mode") == "all":
                    save_mod = 1
                elif self.args.get("inf_save_mode") == "some":
                    save_mod = 100
                
                if batch_idx % save_mod == 0:
                    cmap_gt = matplotlib.cm.get_cmap("inferno_r")
                    cmap_gt.set_bad(color='1')
                    cmap_gt.set_under(color='1')
                    cmap_pred = matplotlib.cm.get_cmap("inferno_r")

                    depth_min = self.args[self.args.basic.dataset].min_depth
                    breakpoint()
                    depth_max = depth_gt.max()

                    # Save things, using batch_idx as base name
                    normed_image = self.unnormalize(batch['image'][0]).cpu()

                    plt.clf()
                    plt.axis('off')
                    plt.imshow(np.transpose(normed_image.detach().cpu().numpy(), (1, 2, 0)))
                    plt.savefig(os.path.join(self.args.predict_output_dir, f"{batch_idx}_im.png"), bbox_inches='tight', dpi=250)
                    
                    # Depth visualisations
                    plt.imshow(depth_gt[0].detach().cpu().numpy().squeeze(0), vmin=depth_min, vmax=depth_max, cmap=cmap_gt)
                    plt.savefig(os.path.join(self.args.predict_output_dir, f"{batch_idx}_depth_gt.png"), bbox_inches='tight', dpi=250)
                    plt.imshow(depth_pred[0].detach().cpu().numpy().squeeze(0), vmin=depth_min, vmax=depth_max, cmap=cmap_pred)
                    plt.savefig(os.path.join(self.args.predict_output_dir, f"{batch_idx}_depth_pred.png"), bbox_inches='tight', dpi=250)
                    
                    # Depth raw values
                    # torch.save(depth_gt[0].detach().cpu(), os.path.join(self.args.predict_output_dir, f"{batch_idx}_depth_gt_raw.pkl"))
                    # torch.save(depth_pred[0].detach().cpu(), os.path.join(self.args.predict_output_dir, f"{batch_idx}_depth_pred_raw.pkl"))
            
            case "none" | _:
                pass    # For readability. 

        # Make a dict of all the metrics + filenames, then add to the prediction dict for later saving
        curr_pred_dict = {
            "batch_idx": batch_idx,
            "image_filename": batch["image_path"][0],
            "depth_gt_filename": batch["depth_path"][0],

            "abs_rel": self.abs_rel.compute().item(),
            "sq_rel": self.sq_rel.compute().item(),
            "rmse": self.rmse.compute().item(),
            "rmse_log": self.rmse_log.compute().item(),
            "log10": self.log10.compute().item(),
            "acc_1": self.acc_1.compute().item(),
            "acc_2": self.acc_2.compute().item(),
            "acc_3": self.acc_3.compute().item(),

            "abs_rel_ra": self.abs_rel_ra.compute().item(),
            "sq_rel_ra": self.sq_rel_ra.compute().item(),
            "rmse_ra": self.rmse_ra.compute().item(),
            "rmse_log_ra": self.rmse_log_ra.compute().item(),
            "log10_ra": self.log10_ra.compute().item(),
            "acc_1_ra": self.acc_1_ra.compute().item(),
            "acc_2_ra": self.acc_2_ra.compute().item(),
            "acc_3_ra": self.acc_3_ra.compute().item(),
            "loss": loss.item(),
        }

        self.prediction_dict[batch_idx] = curr_pred_dict

        # Manually reset every batch as we want to save individual metrics, not batch metrics.
        self.abs_rel.reset()
        self.sq_rel.reset()
        self.rmse.reset()
        self.rmse_log.reset()
        self.log10.reset()
        self.acc_1.reset()
        self.acc_2.reset()
        self.acc_3.reset()

        self.abs_rel_ra.reset()
        self.sq_rel_ra.reset()
        self.rmse_ra.reset()
        self.rmse_log_ra.reset()
        self.log10_ra.reset()
        self.acc_1_ra.reset()
        self.acc_2_ra.reset()
        self.acc_3_ra.reset()

        return loss


    def on_predict_end(self):
        # Save the prediction dict to file
        out_df = pd.DataFrame.from_dict(self.prediction_dict, orient='index')
        out_path = os.path.join(self.args.predict_output_dir, f"prediction_metrics.csv")    
        out_df.to_csv(out_path)


    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Perform data augmentation on GPU (if doing training), and normalize.
        
        Expects batch to be a dict containing keys 'image' and 'depth_gt'.
        Returns a tuple of (image, depth).
        """

        if self.args.basic.get("use_adabins_dataloader") != True:
            # The adabins dataloader takes care of both normalization and data augmentation, so we only do those things if not using it
            if self.trainer.training:
                batch = self.data_augmentation(batch)

            batch['image'] = self.normalize(batch['image']) # Important: Normalise images to ImageNet mean and std.
        
            return batch
        else:
            # If using the adabins dataloader, the keys are different.
            if self.training:
                return batch
            else:
                batch['depth'] = batch['depth'].permute(0, 3, 1, 2).contiguous()
                return batch


    def configure_optimizers(self):
        if "slow_encoder" in self.args[self.args.model.name]:
            params = [
                {"params": self.model.get_encoder_params(), "lr": self.args.optimizer.lr / self.args[self.args.model.name].slow_encoder},
                {"params": self.model.get_non_encoder_params(), "lr": self.args.optimizer.lr},
                # {"params": self.model.get_zero_params(), "lr": 0.0}
            ]
        else:
            params = [
                {"params": self.model.get_encoder_params(), "lr": self.args.optimizer.lr},
                {"params": self.model.get_non_encoder_params(), "lr": self.args.optimizer.lr},
                # {"params": self.model.get_zero_params(), "lr": 0.0}
            ]

        # params = [{"params": self.model.get_encoder_params(), "lr": self.args.optimizer.lr}]

        # Freezing anything that we want frozen
        # Note that if grads need to propagate through then params should have requires_grad=True, but not be added to an optimizer (so they don't update).
        for p in self.model.get_frozen_params():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(params=params, lr=self.args.optimizer.lr, weight_decay=self.args.optimizer.wd)
        if "use_swa" not in self.args.optimizer or ("use_swa" in self.args.optimizer and self.args.optimizer.use_swa):
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                            max_lr=self.args.optimizer.lr,
                            total_steps=self.trainer.estimated_stepping_batches,
                            cycle_momentum=True,
                            base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
                            div_factor=self.args.optimizer.div_factor,
                            final_div_factor=self.args.optimizer.final_div_factor)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }
        else:
            return optimizer

    
    def train_dataloader(self):
        if self.args.basic.get("use_adabins_dataloader") == True:
            train_loader = DepthDataLoader(self.args, 'train').data
        else:
            match self.args.basic.dataset:
                case "nyu":
                    sys.exit("Error: wrong NYUD2 (set args.basic.use_adabins_dataloader instead)")
                    train_dset = NYUD2(self.args, mode='train', transform=self.train_preprocessor)
                case "kitti":
                    sys.exit("Error: wrong KITTI (set args.basic.use_adabins_dataloader instead)")
                    train_dset = KITTI(self.args, mode='train', transform=self.train_preprocessor)
                case _:
                    sys.exit("Error while loading training dataset: unrecognised args.basic.dataset.")

            # Make the dataloader
            train_loader = torch.utils.data.DataLoader(
                train_dset,
                shuffle=True,
                drop_last=False,
                batch_size=self.args.basic.batch_size,
                num_workers=self.args.hardware.num_workers,
                persistent_workers=not(self.args.debug),     # args.debug sets num_workers to 0, which requires persistent_workers=False.
                pin_memory=True,
            )

        return train_loader


    def val_dataloader(self):
        if self.args.basic.get("use_adabins_dataloader") == True:
            val_loader = DepthDataLoader(self.args, 'online_eval').data
        else:
            match self.args.basic.dataset:
                case "nyu":
                    sys.exit("Error: wrong NYUD2 (set args.basic.use_adabins_dataloader instead)")
                    val_dset = NYUD2(self.args, mode='online_eval', transform=self.val_preprocessor)
                case "kitti":
                    sys.exit("Error: wrong KITTI (set args.basic.use_adabins_dataloader instead)")
                    val_dset = KITTI(self.args, mode='online_eval', transform=self.val_preprocessor)
                case _:
                    sys.exit("Error while loading validation dataset: unrecognised args.basic.dataset.")
            
            # Make the dataloader
            val_loader = torch.utils.data.DataLoader(
                val_dset,
                shuffle=False,
                drop_last=False,
                batch_size=self.args.basic.batch_size,
                num_workers=self.args.hardware.num_workers,
                persistent_workers=not(self.args.debug),     # args.debug sets num_workers to 0, which requires persistent_workers=False.
                pin_memory=True,
            )

        return val_loader


    def predict_dataloader(self):
        if self.args.basic.get("use_adabins_dataloader") == True:
            predict_loader = DepthDataLoader(self.args, 'online_eval').data
        else:
            match self.args.basic.dataset:
                case "nyu":
                    val_dset = NYUD2(self.args, mode='online_eval', transform=self.val_preprocessor)
                case "kitti":
                    val_dset = KITTI(self.args, mode='online_eval', transform=self.val_preprocessor)
            predict_loader = torch.utils.data.DataLoader(
                val_dset,
                shuffle=False,
                drop_last=False,
                batch_size=self.args.basic.batch_size,
                num_workers=self.args.hardware.num_workers,
                persistent_workers=not(self.args.debug),     # args.debug sets num_workers to 0, which requires persistent_workers=False.
                pin_memory=True,
            )
        return predict_loader
        
