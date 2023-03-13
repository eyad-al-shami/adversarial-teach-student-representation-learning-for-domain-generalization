from torch.utils.data import DataLoader
import torch
import os
import shutil
import random
import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_readable_date_time():
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M")

def merge_parameters(models):
    """Merge the parameters of the models in the list models into a single list of parameters.
    """
    params = []
    for model in models:
        params += list(model.parameters())
    return params

def set_logger(cfg):
    if (cfg.LOGGING.ENABLED == False):
        return None
    if cfg.LOGGING.LOGGER == "wandb":
        run = wandb.init(project=cfg.LOGGING.WANDB.PROJECT, name=cfg.LOGGING.EXPERIMENT_NAME, config=cfg)
        logger = wandb
    elif cfg.LOGGING.LOGGER == "tensorboard":
        logger = SummaryWriter()
    else:
        # TODO: add logging to file
        logger = None
    return logger
class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
    
def save_checkpoint(state, is_best, checkpoint, name):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, f'{name}.pth')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth'))

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    return checkpoint
class OnlineMeanStd:
    def __init__(self):
        pass

    def __call__(self, dataset, batch_size, method='strong'):
        """
        Calculate mean and std of a dataset in lazy mode (online)
        On mode strong, batch size will be discarded because we use batch_size=1 to minimize leaps.
        :param dataset: Dataset object corresponding to your dataset
        :param batch_size: higher size, more accurate approximation
        :param method: weak: fast but less accurate, strong: slow but very accurate - recommended = strong
        :return: A tuple of (mean, std) with size of (3,)
        """

        if method == 'weak':
            loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=0)
            mean = 0.
            std = 0.
            nb_samples = 0.
            for data in loader:
                data = data['y_descreen']
                batch_samples = data.size(0)
                data = data.view(batch_samples, data.size(1), -1)
                mean += data.mean(2).sum(0)
                std += data.std(2).sum(0)
                nb_samples += batch_samples

            mean /= nb_samples
            std /= nb_samples

            return mean, std

        elif method == 'strong':
            loader = DataLoader(dataset=dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=0)
            cnt = 0
            fst_moment = torch.empty(3)
            snd_moment = torch.empty(3)

            for data in loader:
                data, label, domain = data
                b, c, h, w = data.shape
                nb_pixels = b * h * w
                sum_ = torch.sum(data, dim=[0, 2, 3])
                sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
                fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
                snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

                cnt += nb_pixels

            return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

class EMAWeight:
    def __init__(self, start_value, final_value):
        '''
        start_value: the value to start at
        final_value: the value to end at
        Example: Decay(0.999999, 0.999)
        '''
        if (start_value < final_value):
            raise ValueError("start_value must be >= final_value")
        self.value = str(start_value)
        self.final_value = str(final_value)
        self.count = 0

    def get_value(self):
        if (self.count == 0):
            self.count -= 1
            return float(self.value)
        if (self.value[:self.count] == self.final_value):
            return float(self.value[:self.count])
        returned_value = float(self.value[:self.count])
        self.count -= 1
        return returned_value

@torch.no_grad()
def copy_model_weights(from_model, to_model):
    for param1, param2 in zip(from_model.parameters(), to_model.parameters()):
        param2.data.copy_(param1.data)

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

plt.rcParams["savefig.bbox"] = 'tight'

def show_batch(batch, dataset):
    fig, ax = plt.subplots(1, batch[0].shape[0], figsize=(20, 20))
    for i in range(batch[0].shape[0]):
        ax[i].imshow(batch[0][i].permute(1, 2, 0), interpolation='nearest')
        ax[i].set_title(dataset.index_to_class[batch[2][i].item()])
        ax[i].axis('off')
    plt.show()