from .defaults import _C as cfg_default
import utils
import torch
import logging

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg_default.clone()

def merge_args_cfg(cfg, args):
    # TODO logging must always be enabled, use tensorboard by default
    if args.logger:
        cfg.LOGGING.LOGGER = args.logger
        # If the experiment name is not provided, then use the date and time
        if (args.experiment_name and args.experiment_name.lower() == "hp"):
            print("CHANGE EXPERIMENT NAME")
            cfg.LOGGING.EXPERIMENT_NAME = f"A: {cfg.MODEL.AUGMENTER.LR} - TAU: {cfg.MODEL.TEACHER.TAU} - TW: {cfg.MODEL.TEACHER.WARMUP_LR} - S: {cfg.MODEL.STUDENT.LR} - {utils.get_readable_date_time()}"
        else:
            cfg.LOGGING.EXPERIMENT_NAME = args.experiment_name if args.experiment_name else utils.get_readable_date_time()

    if args.data_dir:
        cfg.DATASET.ROOT = args.data_dir

    cfg.OUTPUT_DIR = args.output_dir if args.output_dir else cfg.LOGGING.EXPERIMENT_NAME
    # remove andy characters from the output directory name that may cause problems
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace(":", "=")
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace(" ", "_")
    invalid = '<>:"/\|?*'
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.translate({ord(c): None for c in invalid})



    # if args.restore_file:
    #     cfg.RESUME = args.restore_file

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.dataset:
        cfg.DATASET.NAME = args.dataset

    if  args.use_cuda:
        # check if cuda is available
        if torch.cuda.is_available():
          cfg.USE_CUDA = args.use_cuda
        else:
          logging.warning("CUDA is not available, using CPU instead")
          cfg.USE_CUDA = False
          cfg.DEVICE = "cpu"
    
    if (args.config_debug):
        cfg.CONFIG_DEBUG = True

    if cfg.MODEL.AUGMENTER.COMPUTE_MARGIN:
        # assert that the length of the specified domains is more than 1
        assert len(cfg.DATASET.SOURCE_DOMAINS) > 1, "The length of the source domains must be more than 1 if the margin is computed"

    if args.dry_run:
        cfg.DRY_RUN = True

    if args.use_cpu:
        cfg.USE_CUDA = False
        cfg.DEVICE = "cpu"

    
    cfg.merge_from_list(args.opts)

def setup_cfg(args):
    cfg = get_cfg_defaults()
    
    # 1. From experiment config file
    if args.experiment_cfg:
        cfg.merge_from_file(args.experiment_cfg)

    # 2. From input arguments
    merge_args_cfg(cfg, args)
    # 4. From optional input arguments
    # cfg.merge_from_list(args.opts)
    # clean_cfg(cfg, args.trainer)
    
    cfg.freeze()
    return cfg
