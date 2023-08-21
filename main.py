#!/usr/bin/env python
# main.py
# Base script to invoke for everything

import os, sys
import re
import argparse, argcomplete
import logging
from omegaconf import OmegaConf

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from modules.MainRunnerLM import MainRunnerLM

import misc_utils


def main(args):
    logger = logging.getLogger(__name__)

    # Define model (pytorch lightning module that contains model definition)
    model = MainRunnerLM(args)

    if args.basic.get("from_checkpoint") is not None:
        logger.info(f"Loading from checkpoint: {args.basic.from_checkpoint}")
        model = model.load_from_checkpoint(checkpoint_path=args.basic.from_checkpoint, args=args)

    callbacks_list = []

    # For monitoring the learning rate as determined by the scheduler
    callbacks_list.append(pl.callbacks.LearningRateMonitor(logging_interval='step'))
    
    if "gradient_clip_val" in args.optimizer:
        gradient_clip_val = args.optimizer.gradient_clip_val
    else:
        gradient_clip_val = 0

    # If set, remove 1-cycle LR scheduler and use Stochastic Weight Averaging.
    if "use_swa" in args.optimizer and args.optimizer.use_swa:
        logger.info("Using StochasticWeightAveraging")
        callbacks_list.append(pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2))
    
    tb_writer = pl.loggers.TensorBoardLogger(
        save_dir=args.paths.run_dir,
        name=args.basic.name,
        default_hp_metric=False,
        max_queue=1,
        )

    if args.get("validate"):
        logger.info("==== RUNNING VALIDATION ====")
        assert args.basic.get("val_checkpoint") is not None, "Error: no validation checkpoint set."
        logger.info(f"Checkpoint file used: {args.basic.val_checkpoint}")
        
        validation_output_filename = "validation_output.txt"
        if args.get("override_target_dataset") is not None:
            logger.critical(f"""====== Dataset override in effect!: {args.get("override_target_dataset")}""")
            args.basic.dataset = args.get("override_target_dataset")
            args.basic.use_adabins_dataloader = True

            validation_output_filename = f"validation_output_zeroshot_{args.basic.dataset}.txt"

        args.basic.batch_size = 1   # Override batch size
        trainer = pl.Trainer(
            limit_train_batches=1 if args.debug else None,
            limit_val_batches=1 if args.debug else None,
            max_epochs=1 if args.debug else args.basic.max_epochs,
            check_val_every_n_epoch=args.basic.validate_every,
            gradient_clip_val=gradient_clip_val,
            accelerator="auto",
            strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True, static_graph=False),
            # strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=False, static_graph=True),
            devices=1,
            # logger=tb_writer,
            log_every_n_steps=1 if args.debug else 50,
            callbacks=callbacks_list,
            enable_model_summary=False
        )

        # fix the broken state dict.
        tmp_ckpt = torch.load(args.basic.val_checkpoint)
        tmp_ckpt['state_dict'] = {re.sub(r"\.embeddings\.weight", "", k) if re.compile(".*extra_tkns_learnables_.*\.embeddings\.weight").match(k) else k: v for k, v in tmp_ckpt['state_dict'].items()}
        tmp_ckpt['state_dict'].pop("data_augmentation.random_planckian_jitter.pl", None)
        torch.save(tmp_ckpt, "_tmp_ckpt.ckpt")

        val_output = trainer.validate(
            model=model,
            verbose=True,
            # ckpt_path=args.basic.val_checkpoint,
            ckpt_path="_tmp_ckpt.ckpt"
        )
        
        with open(os.path.join(args.val_output_dir, validation_output_filename), 'w') as f:
            f.write(f"{args.basic.name}\n")
            f.write(str(val_output))
            log_str = f"\nabs_rel, sq_rel, rms, rmsl, log10, d1, d2, d3:  \n{val_output[0]['metrics/abs_rel']}, {val_output[0]['metrics/sq_rel']}, {val_output[0]['metrics/rmse']}, {val_output[0]['metrics/rmse_log']}, {val_output[0]['metrics/log10']}, {val_output[0]['metrics/acc_1']}, {val_output[0]['metrics/acc_2']}, {val_output[0]['metrics/acc_3']}  \n ==#==  \nabs_rel_ra, sq_rel_ra, rms_ra, rmsl_ra, log10_ra, d1_ra, d2_ra, d3_ra:  \n{val_output[0]['metrics_ra/abs_rel_ra']}, {val_output[0]['metrics_ra/sq_rel_ra']}, {val_output[0]['metrics_ra/rmse_ra']}, {val_output[0]['metrics_ra/rmse_log_ra']}, {val_output[0]['metrics_ra/log10_ra']}, {val_output[0]['metrics_ra/acc_1_ra']}, {val_output[0]['metrics_ra/acc_2_ra']}, {val_output[0]['metrics_ra/acc_3_ra']}"
            f.write(log_str)

        print(str(val_output))
        print(log_str)
    
    elif args.get("inference"):
        logger.info("==== RUNNING INFERENCE ====")
        assert args.basic.get("val_checkpoint") is not None, "Error: no validation checkpoint set."
        logger.info(f"Checkpoint file used: {args.basic.val_checkpoint}")
        if args.get("override_target_dataset") is not None:
            sys.exit("ERROR: dataset override not implemented for inference yet.")
            logger.critical(f"""====== Dataset override in effect!: {args.get("override_target_dataset")}""")
            args.basic.dataset = args.get("override_target_dataset")
            inference_output_filename = f"validation_output_zeroshot_{args.basic.dataset}.txt"
        
        args.basic.batch_size = 1   # Override batch size.
        trainer = pl.Trainer(
            limit_train_batches=1 if args.debug else None,
            limit_val_batches=1 if args.debug else None,
            limit_predict_batches=1 if args.debug else None,
            max_epochs=1 if args.debug else args.basic.max_epochs,
            check_val_every_n_epoch=args.basic.validate_every,
            gradient_clip_val=gradient_clip_val,
            accelerator="auto",
            strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True, static_graph=False),
            devices=1,
            logger=tb_writer,
            log_every_n_steps=1 if args.debug else 50,
            callbacks=callbacks_list,
            enable_model_summary=False
        )
        
        predictions = trainer.predict(
            model=model,
            # ckpt_path=args.basic.val_checkpoint,
            # ckpt_path=None
        )
        logger.info(f"Done, results and metrics saved to {args.predict_output_dir}")
    
    elif args.get("no_train_validation"):
        # For instantiating and immediately validating a model without any training at all.
        logger.info("==== RUNNING VALIDATION WITH **NO CHECKPOINT** ====")

        # Override prediction output directories here:
        args.predict_output_dir = os.path.join(tb_writer.log_dir, "predict_output_no_train")
        if not os.path.exists(args.predict_output_dir):
            os.makedirs(args.predict_output_dir)

        args.basic.batch_size = 1   # Override batch size.
        trainer = pl.Trainer(
            limit_train_batches=1 if args.debug else None,
            limit_val_batches=1 if args.debug else None,
            limit_predict_batches=1 if args.debug else None,
            max_epochs=1 if args.debug else args.basic.max_epochs,
            check_val_every_n_epoch=args.basic.validate_every,
            gradient_clip_val=gradient_clip_val,
            accelerator="auto",
            strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True, static_graph=False),
            devices=1,
            logger=tb_writer,
            log_every_n_steps=1 if args.debug else 50,
            callbacks=callbacks_list,
            enable_model_summary=False
        )
        
        # predictions = trainer.predict(
        #     model=model,
        # )
        # logger.info(f"Done, results and metrics saved to {args.predict_output_dir}")
        val_output = trainer.validate(
            model=model,
            verbose=True,
            # ckpt_path=args.basic.val_checkpoint,
        )
        
        with open(os.path.join(args.predict_output_dir, "validation_output.txt"), 'w') as f:
            f.write(args.basic.name)
            f.write(str(val_output))
            log_str = f"\nabs_rel, sq_rel, rms, rmsl, log10, d1, d2, d3:  \n{val_output[0]['metrics/abs_rel']}, {val_output[0]['metrics/sq_rel']}, {val_output[0]['metrics/rmse']}, {val_output[0]['metrics/rmse_log']}, {val_output[0]['metrics/log10']}, {val_output[0]['metrics/acc_1']}, {val_output[0]['metrics/acc_2']}, {val_output[0]['metrics/acc_3']}  \n ==#==  \nabs_rel_ra, sq_rel_ra, rms_ra, rmsl_ra, log10_ra, d1_ra, d2_ra, d3_ra:  \n{val_output[0]['metrics_ra/abs_rel_ra']}, {val_output[0]['metrics_ra/sq_rel_ra']}, {val_output[0]['metrics_ra/rmse_ra']}, {val_output[0]['metrics_ra/rmse_log_ra']}, {val_output[0]['metrics_ra/log10_ra']}, {val_output[0]['metrics_ra/acc_1_ra']}, {val_output[0]['metrics_ra/acc_2_ra']}, {val_output[0]['metrics_ra/acc_3_ra']}"
            f.write(log_str)

        print(str(val_output))
        print(log_str)
    
    elif args.get("no_train_inference"):
        # For instantiating and immediately validating a model without any training at all.
        logger.info("==== RUNNING INFERENCE WITH **NO CHECKPOINT** ====")

        # Override prediction output directories here:
        args.predict_output_dir = os.path.join(tb_writer.log_dir, "predict_output_no_train")
        if not os.path.exists(args.predict_output_dir):
            os.makedirs(args.predict_output_dir)

        args.basic.batch_size = 1   # Override batch size.
        trainer = pl.Trainer(
            limit_train_batches=1 if args.debug else None,
            limit_val_batches=1 if args.debug else None,
            limit_predict_batches=1 if args.debug else None,
            max_epochs=1 if args.debug else args.basic.max_epochs,
            check_val_every_n_epoch=args.basic.validate_every,
            gradient_clip_val=gradient_clip_val,
            accelerator="auto",
            strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True, static_graph=False),
            devices=1,
            logger=tb_writer,
            log_every_n_steps=1 if args.debug else 50,
            callbacks=callbacks_list,
            enable_model_summary=False,
        )
        
        predictions = trainer.predict(
            model=model,
        )
        logger.info(f"Done, results and metrics saved to {args.predict_output_dir}")

    else:
        logger.info("==== RUNNING TRAIN/VAL LOOP ====")

        # Define pytorch lightning trainer
        # Checkpointing behaviours
        callbacks_list.append(pl.callbacks.ModelCheckpoint(monitor="metrics/abs_rel", save_last=True, save_top_k=1, mode="min"))
        callbacks_list.append(pl.callbacks.ModelSummary(max_depth=3))

        trainer = pl.Trainer(
            limit_train_batches=5 if args.debug else None,
            limit_val_batches=1 if args.debug else None,
            max_epochs=5 if args.debug else args.basic.max_epochs,
            check_val_every_n_epoch=args.basic.validate_every,
            gradient_clip_val=gradient_clip_val,
            accelerator="auto",
            strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True, static_graph=False),
            # strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=False, static_graph=True),
            devices=args.devices,
            logger=tb_writer,
            log_every_n_steps=1 if args.debug else 50,
            callbacks=callbacks_list,
            # enable_model_summary=True,
        )
        trainer.fit(model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config_file", required=True, type=argparse.FileType('r', encoding='UTF-8'), help="Path to the config/params YAML file.")
    
    # Running mode
    parser.add_argument("-v", "--validate", action="store_true", help="""
        Runs validation using the latest available checkpoint that shares a name with the params file, unless the checkpoint file is specified by the
        params file in args.basic.val_checkpoint, in which case that checkpoint is evaluated instead. Uses only one device and a batch size of 1.
        Can also be used on the hparams.yaml automatically saved in each experiment's run directory (as args.basic.name will be present in this file).
        """)
    parser.add_argument("-i", "--inference", action="store_true", help="Run inference (like validation but with bigger batches and no saved metrics file)")
    parser.add_argument("--no_train_validation", action="store_true", help="""
        Instantiate the model, then immediately run validation and take aggregate metrics from it with no training at all.
        To use this, args.basic.val_checkpoint must be unset in the params file.""")
    parser.add_argument("--no_train_inference", action="store_true", help="""
        Instantiate the model, then immediately run inference and take metrics from each example with no training at all.
        This is used to save examples to make into figures or to get per-example metrics.
        To use this, args.basic.val_checkpoint must be unset in the params file.""")

    # Inference-like mode arguments
    parser.add_argument("--inf_save_mode", default="all", nargs="?", choices=["all", "some", "none"], help="""
        Only used when doing inference or no_train_validation. Whether to save all qualitative samples, 
        only some of them (every 100), or none at all. Quantitative results are always saved.
        """)

    # Some debugging options
    parser.add_argument("--debug", action="store_true", help="""
        Sets debug mode. Force single-device training with no spawned dataloader workers, to allow breakpoints to work.
        Also forces maximum 50 training batches, for speed of debugging.
        """)
    parser.add_argument("--log_debug", action="store_true", help="""
        If set sets log level to logging.DEBUG. Separate from --debug because sometimes debug output isn't helpful.
        """)

    parser.add_argument("--override_target_dataset", type=str, default=None, help="""
        If set, will force-override the dataset used, and will also modify the output dir name to match the new dataset. 
        This exists for zero-shot domain transfer experiments:
        Checkpoint loading logic is based on param file names, so having this option avoids having to remake a bunch of params files.
        ONLY works for --validate and --inference.
        """)
    
    argcomplete.autocomplete(parser)
    cl_args = parser.parse_args()

    # Parse args
    args = OmegaConf.load(cl_args.config_file)
    if "args" in args:
        args = args.args    # This is to allow loading of auto-saved hparams.yaml files
    args.config_file = cl_args.config_file.name
    args.debug = cl_args.debug
    args.log_debug = cl_args.log_debug
    args.validate = cl_args.validate
    args.inference = cl_args.inference
    args.no_train_validation = cl_args.no_train_validation
    args.no_train_inference = cl_args.no_train_inference
    args.inf_save_mode = cl_args.inf_save_mode
    args.override_target_dataset = cl_args.override_target_dataset

    assert not (args.get("validate") and args.get("inference"))

    # Set up params for the debug mode (1 device, don't spawn extra workers in dataloader to let breakpoint() still work)
    if args.debug:
        logging.info("Debug mode active (--debug)")
    args.devices = 1 if args.debug or args.validate or args.inference else None
    args.hardware.num_workers = 0 if args.debug else args.hardware.num_workers

    # Handle overrides and defaults, do some checking
    args = misc_utils.check_and_validate_args(args)

    logging.basicConfig(level=logging.DEBUG if args.log_debug else logging.INFO, force=True, format="[%(levelname)s][%(name)s] %(message)s")
    logging.info("Starting")
    logging.debug("Debug log active")
    logging.debug(args)

    pl.seed_everything(42, workers=True)

    main(args)