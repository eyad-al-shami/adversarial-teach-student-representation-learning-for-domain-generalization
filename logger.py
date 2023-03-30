
import wandb
import os
import sys
import time
import os.path as osp
from torch.utils.tensorboard import SummaryWriter

__all__ = ["Logger"]

class Logger():
    def __init__(self, config):
        self.config = config
        self.logger_used = "tensorboard"

        # check the LOGGING.LOG_DIR
        if not osp.exists(config.LOGGING.LOG_DIR):
            os.makedirs(config.LOGGING.LOG_DIR)

        # create a directory for the experiment
        if not osp.exists(osp.join(config.LOGGING.LOG_DIR, config.LOGGING.EXPERIMENT_NAME)):
            os.makedirs(osp.join(config.LOGGING.LOG_DIR, config.LOGGING.EXPERIMENT_NAME))

        if not config.LOGGING.LOGGER:
            print("No logger is used, using tensorboard as a local logger")
            self.writer = SummaryWriter(log_dir=osp.join(config.LOGGING.LOG_DIR, config.LOGGING.EXPERIMENT_NAME))
        else:
            if config.LOGGING.LOGGER == "wandb":
                self.writer = wandb
                self.writer.init(project=config.LOGGING.PROJECT, name=config.LOGGING.EXPERIMENT_NAME, config=config)
                self.logger_name = "wandb"
            elif config.LOGGING.LOGGER == "tensorboard":
                self.writer = SummaryWriter(log_dir=osp.join(config.LOGGING.LOG_DIR, config.LOGGING.EXPERIMENT_NAME))

    def log(self, metrics, step):
        if self.logger_used == "wandb":
            self.writer.log(metrics, step=step)
        elif self.logger_used == "tensorboard":
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

