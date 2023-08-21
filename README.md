# PromptLearningCLIP
This folder contains the code for the paper "Learning to Prompt CLIP for Monocular Depth Estimation: Exploring the Limits of Human Language", Dylan Auty and Krystian Mikolajczyk, ICCV Workshop on Open Vocabulary Scene Understanding (OpenSUN3D) 2023.

## Installation
Create a new conda environment from `conda_environment_files/environment.yaml`. Activate it with `conda activate promptlearningclip`.

## Params files
Params are handled using OmegaConf, and defined via yaml files. The most up-to-date set of parameters and descriptive comments for them can be found in `params/basicParams.yaml`, which is used during development. To define a new experiment, please refer to that file for instructions on what each of the parameters do.

## Datasets
Preparation of datasets should be done following [the official BTS repo](https://github.com/cleinc/bts) and the Pytorch-specific instructions linked within there.

All datasets are assumed to be located in `./data`, e.g. `./data/nyu` or `./data/kitti`. This can be changed in the parameter yaml file if needed.

## Training
Define a new parameter file, then run:
```python
python main.py -c /path/to/params/file.yaml
```

Results are written to `./runs` by default, and are in Tensorboard format.

For debugging (0 workers for dataloader, running on only one GPU, small number of iterations for both training and testing), the `--debug` command line flag can be optionally added.

## Validation
Use the `-v` flag. Specify either the params file or the automatically-saved `hparams.yaml`:
1. Using regular params file: code will attempt to find the most recently-modified file called `last.ckpt` in the run directory corresponding to the name of the params file (or `args.basic.name` if set in the params file). This is normally fine, but if there are multiple versions of the experiment (i.e. `run_name/version_0`, `run_name/version_1` etc.) then this may not behave as expected.
2. Using auto-saved `hparams.yaml`: Will find the most recently-modified checkpoint called `last.ckpt` within the same directory that the `hparams.yaml` file is located in. File **must** be named `hparams.yaml` for this to work.

Validation mode runs on only one device, with a batch size of 1. It will save a file called `validation_output.txt` in the run directory, containing two sets of metrics: the image-wise running average, following the formulation used in BTS and AdaBins implementations, and the pixelwise total average across the entire validation set. The former is what is reported in the paper, to facilitate comparison with other methods in the literature.
