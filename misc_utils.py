# utils.py
# Contains various utility functions.

import os, sys, logging
import re
from pathlib import Path
import glob
from omegaconf import OmegaConf


def check_and_validate_args(args):
    logger = logging.getLogger(__name__)
    """Runs validation on the args passed via the params YAML
    """
    if "name" not in args.basic or args.basic.name in [None, "None", "none", ""]:
        args.basic.name = os.path.splitext(os.path.basename(args.config_file))[0]
    
    # Some non-exhaustive error checking
    assert all([k in args.keys() for k in ["basic", "paths", "loss", "hardware"]])
    assert args.basic.dataset in args.keys()

    if args.get("no_train_validation"):
        # For use when the model is to be instantiated, then validation immediately run and results saved.
        assert args.basic.get("val_checkpoint") is None, "Error: validation checkpoint will be ignored because --no_train_validation is set."
        
        # if os.path.basename(args.config_file) == "hparams.yaml":
        #     # In this case, we'll override the output directory for the validation results to be in the same dir
        #     # as the hparams.yaml file. This is to facilitate scriptable evaluation of a whole run dir.
        #     args.val_output_dir = os.path.dirname(args.config_file)
        # else:
        #     args.val_output_dir = os.path.dirname(os.path.dirname(args.basic.val_checkpoint))

        # args.predict_output_dir = os.path.join(args.val_output_dir, "predict_output_no_train")
        # if not os.path.exists(args.predict_output_dir):
        #     os.makedirs(args.predict_output_dir)

        # A kludgy fix for bad dataset params - this is OK because those params never change.
        override_args = OmegaConf.load("params/basicParams.yaml")
        logger.critical("===== ************** WARNING ************** =====")
        logger.critical("===== ARG OVERRIDE FOR NYU AND KITTI IN USE =====")
        args.nyu = override_args.nyu
        args.kitti = override_args.kitti

    elif args.get("validate") or args.get("inference"):
        if args.basic.get("val_checkpoint") is None:
            if os.path.basename(args.config_file) == "hparams.yaml":
                args.basic.val_checkpoint = get_latest_checkpoint(args, dir=os.path.dirname(args.config_file))
            else:
                args.basic.val_checkpoint = get_latest_checkpoint(args)

        if os.path.basename(args.config_file) == "hparams.yaml":
            # In this case, we'll override the output directory for the validation results to be in the same dir
            # as the hparams.yaml file. This is to facilitate scriptable evaluation of a whole run dir.
            args.val_output_dir = os.path.dirname(args.config_file)
        else:
            args.val_output_dir = os.path.dirname(os.path.dirname(args.basic.val_checkpoint))

        if args.get("inference"):
            args.predict_output_dir = os.path.join(args.val_output_dir, "predict_output")
            if not os.path.exists(args.predict_output_dir):
                os.makedirs(args.predict_output_dir)

        # A kludgy fix for bad dataset params - this is OK because those params never change.
        override_args = OmegaConf.load("params/basicParams.yaml")
        logger.critical("===== ************** WARNING ************** =====")
        logger.critical("===== ARG OVERRIDE FOR NYU AND KITTI IN USE =====")
        args.nyu = override_args.nyu
        args.kitti = override_args.kitti

    return args


def remove_leading_slash(s):
    """Removes the leading slash from a string. Needed because the NYUD2 file path names come
    with leading slashes, and without this fn they would be interpreted as complete paths on their own.
    """
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


def get_latest_checkpoint(args, dir=None):
    """ Finds the latest checkpoint that matches the name of the params file. 
    If dir is not None, it will search that dir. Otherwise, it will search the run_dir (which may not include the /version_number sub-directories
    that are created when running training multiple times with the same name."""
    dir_to_check = dir if dir is not None else os.path.join(args.paths.run_dir, args.basic.name)
    candidate_chkpts = [path for path in filter(os.path.isfile, Path(dir_to_check).rglob(f"*last.ckpt"))] 
    if len(candidate_chkpts) < 1:
        print("Error: no checkpoints found for this parameter file. Using None")
        most_recent_chkpt = None
    else:
        most_recent_chkpt = str(max(candidate_chkpts, key=os.path.getctime))
    
    return most_recent_chkpt