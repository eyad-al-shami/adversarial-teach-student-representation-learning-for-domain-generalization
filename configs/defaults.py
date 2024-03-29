# my_project/config.py
from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()
_C.USE_CUDA = True
_C.DEVICE = "cuda" if _C.USE_CUDA else "cpu"
# Set seed to negative value to randomize everything
# Set seed to positive value to use a fixed seed
_C.SEED = 1337
# Print detailed information
# E.g. trainer, dataset, and backbone
_C.VERBOSE = True
_C.DRY_RUN = False
_C.CONFIG_DEBUG = False
_C.OUTPUT_DIR = "saved_models"

###########################
# System Hardware
###########################

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 2

###########################
# Training
###########################

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 60
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.PRINT_FREQ = 100


###########################
# Logging
###########################

_C.LOGGING = CN()

_C.LOGGING.EXPERIMENT_NAME = ""
_C.LOGGING.LOG_DIR = "logs"
_C.LOGGING.LOGGER = None
_C.LOGGING.PROJECT = "Adversarial-TS-DG"


###########################
# Models
###########################
_C.MODEL = CN()
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE_OUT_DIM = 512

_C.MODEL.STUDENT = CN()
_C.MODEL.STUDENT.BACKBONE_NAME = "resnet18"
_C.MODEL.STUDENT.LR = 0.0005
_C.MODEL.STUDENT.LR_DECAY = 0.1
_C.MODEL.STUDENT.LR_DECAY_EPOCH = 30
_C.MODEL.STUDENT.PRETRAINED = True

_C.MODEL.TEACHER = CN()
_C.MODEL.TEACHER.BACKBONE_NAME = "resnet18"
_C.MODEL.TEACHER.TAU = 0.9999
_C.MODEL.TEACHER.WARMUP_EPOCHS = 10
_C.MODEL.TEACHER.PRETRAINED = True
_C.MODEL.TEACHER.WARMUP_LR = 0.0005
_C.MODEL.TEACHER.UPDATE_EVERY = 5
_C.MODEL.TEACHER.LEARNING_RATE_LAMBDA = 0.65
# _C.MODEL.TEACHER.UPDATE_SPECIFIC_LAYERS = False
# _C.MODEL.TEACHER.ONLY_UPDATE_LAYERS = ['net.conv1', 'net.bn1', 'net.layer1']
# _C.MODEL.TEACHER.UPDATE_TEACHER = True
# _C.MODEL.TEACHER.UPDATE_AFTER = 20

_C.MODEL.AUGMENTER = CN()
_C.MODEL.AUGMENTER.LR = 0.05
_C.MODEL.AUGMENTER.NORM_LAYER = "IN"
_C.MODEL.AUGMENTER.GTX = True
_C.MODEL.AUGMENTER.COMPUTE_MARGIN = True # Otherwise, use fixed value of 0.1



###########################
# Dataset
###########################

_C.DATASET = CN()
_C.DATASET.DOMAINS = None
_C.DATASET.CLASSES = None
_C.DATASET.NORMALIZE = None
_C.DATASET.MEAN = None
_C.DATASET.STD = None
_C.DATASET.ROOT = ""
_C.DATASET.SOURCE_DOMAINS = None
_C.DATASET.TARGET_DOMAINS = None
_C.DATASET.NAME = ""

###########################
# Metrics
###########################

_C.METRICS = CN()
_C.METRICS.MONITOR = ["accuracy", "loss"]
_C.METRICS.ACCURACY = CN()
_C.METRICS.ACCURACY.TOP_K = (1,)
_C.METRICS.ACCURACY.TASK = "multiclass"



# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`