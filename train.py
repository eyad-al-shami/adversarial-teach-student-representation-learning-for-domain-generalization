"""Train the model"""
from commandline.cmdParser import parser
import logging
import os
from configs import setup_cfg
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter
import utils
import model.net as net
from data import get_dataset
import torchmetrics
# from evaluate import evaluate

def create_componenets(cfg):
    """
        create the componenets of the learning framework
        1. The augmenter model.
        2. The Teacher model.
        3. The student model.
        4. The classification layer.
    """
    augmenter = net.build_augmenter()
    teacher = net.BackBone(cfg=cfg, component='teacher')
    student = net.BackBone(cfg=cfg, component='student')
    classifier = net.ClassifierLayer(cfg=cfg)
    return augmenter, teacher, student, classifier

def training_validation_loop(cfg, m_monitor, logger):
    augmenter, teacher, student, classifier = create_componenets(cfg)
    # 1. Warmup
    warmup(cfg, teacher, classifier, m_monitor, logger)

def warmup(cfg, backbone, classifier, m_monitor, logger):
    phase = 'warmup'
    # train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.DATASET.CLASSES)
    # train_loss = torchmetrics.MeanMetric()

    backbone.unfreeze()
    classifier.unfreeze()
    backbone.train()
    classifier.train()
    warmup_model = torch.nn.Sequential(backbone, classifier)
    # move to GPU if available
    if (cfg.USE_CUDA):
        warmup_model = warmup_model.to(cfg.DEVICE)
        for m in m_monitor.metrics.values():
            m.to(cfg.DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(warmup_model.parameters(), lr=cfg.MODEL.TEACHER.WARMUP_LR, momentum=0.9)

    _, train_loader = get_dataset(cfg, phase)

    for epoch in range(cfg.MODEL.TEACHER.WARMUP_EPOCHS):
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, domain, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                if (cfg.USE_CUDA):
                    data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                optimizer.zero_grad()
                output = warmup_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    acc = m_monitor.metrics["acc"](output, target)
                    m_monitor.metrics["loss"](loss)
                tepoch.set_postfix(loss=loss.item(), acc=acc.item())
            logger.log({"warmup_loss": m_monitor.metrics["loss"].compute(), "warmup_acc": m_monitor.metrics["acc"].compute(), "epoch": epoch, "phase": phase})
        if cfg.DEBUG:
            break
    m_monitor.reset()
    return warmup_model[0], warmup_model[1]

def set_logger(cfg):
    if cfg.LOGGING.WANDB.ENABLE:
        run = wandb.init(project=cfg.LOGGING.WANDB.PROJECT, name=cfg.LOGGING.EXPERIMENT_NAME, config=cfg)
        logger = wandb
    elif cfg.LOGGING.TENSORBOARD.ENABLE:
        logger = SummaryWriter()
    return logger

def representation_learning():
    pass

def domain_augmentation():
    pass

def validation():
    pass


if __name__ == '__main__':

    # python train.py --experiment_cfg Experiments/E1_No_Normalization.yaml --use_cuda --use_wandb --experiment_name "testing the warmup process"
    # python train.py --experiment_cfg Experiments/E1_No_Normalization.yaml --use_wandb --experiment_name "testing the warmup process"
    
    args = parser.parse_args()
    cfg = setup_cfg(args)
    # Set the random seed for reproducible experiments
    utils.set_random_seed(cfg.SEED)

    print(cfg)
    logger = set_logger(cfg)
    m_monitor = net.Metrics_Monitor(cfg)
    training_validation_loop(cfg, m_monitor, logger)



    # # Set the logger
    # utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # # Create the input data pipeline
    # logging.info("Loading the datasets...")

    # # fetch dataloaders
    # dataloaders = data_loader.fetch_dataloader(
    #     ['train', 'val'], args.data_dir, params)
    # train_dl = dataloaders['train']
    # val_dl = dataloaders['val']

    # logging.info("- done.")

    # # Define the model and optimizer
    # model = net.Net(params).cuda() if params.cuda else net.Net(params)
    # optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # # fetch loss function and metrics
    # loss_fn = net.loss_fn
    # metrics = net.metrics

    # # Train the model
    # logging.info("Starting training for {} epoch(s)".format(params.num_epochs))

    # cfg.merge_from_list(opts)



