import argparse
import utils

parser = argparse.ArgumentParser()

parser.add_argument(
    '--experement_name', 
    default=None, 
    help="Directory containing the dataset")

parser.add_argument(
    '--data_dir', 
    default=None, 
    help="Directory containing the dataset")

parser.add_argument(
    '--output_dir', 
    default=None, 
    help="Directory to save outputs")

parser.add_argument(
    '--restore_file', 
    default=None,
    help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'

parser.add_argument(
    '--seed', 
    default=1337,
    type=int,
    help="Random seed")

parser.add_argument(
    '--source_domains', 
    nargs='+',
    default = ["art_painting", "cartoon", "photo"], 
    type=str)

parser.add_argument(
    '--target_domains', 
    nargs='+', 
    default = ["sketch"],
    type=str)

parser.add_argument(
    '--backbone',
    default='resnet18', 
    type=str)

parser.add_argument(
    '--experiment_cfg',
    default=None,
    type=str)

# set the default name of the run to be the date and time
parser.add_argument(
    '--run_name',
    default=utils.get_readable_date_time(),
    type=str)

# add argument for using wandb, if not provided, then use tensorboard
parser.add_argument(
    '--use_wandb',
    action='store_true',
    help="Use wandb for logging")

parser.add_argument(
    '--use_cuda',
    action='store_true',
    help="Use cuda for training")

parser.add_argument(
    '--dataset', 
    default=None,
    choices=["PACS"],
    type=str, 
    help="The name of the dataset")

# add argument to do one pass to check if the code is working
parser.add_argument(
    '--debug',
    action='store_true',
    help="Do one pass to check if the code is working")