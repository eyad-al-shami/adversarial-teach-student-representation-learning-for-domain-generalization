"""Train the model"""
from commandline.cmdParser import parser
import logging
import os
from configs import setup_cfg
import torch
import torch.optim as optim
from tqdm import tqdm

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
    

    phase = 'training'
    _, train_loader = get_dataset(cfg, phase)

    # move modules to GPU if available
    if (cfg.USE_CUDA):
        augmenter = augmenter.to(cfg.DEVICE)
        teacher = teacher.to(cfg.DEVICE)
        student = student.to(cfg.DEVICE)
        classifier = classifier.to(cfg.DEVICE)
        for m in m_monitor.metrics.values():
            m.to(cfg.DEVICE)
    
    # 1. Warmup
    teacher, classifier = warmup(cfg, teacher, classifier, m_monitor, logger)

    # 2 and 3 Representation Learning and Distillation
    for epoch in range(cfg.MODEL.TEACHER.WARMUP_EPOCHS):
        # reset the metrics
        for m in m_monitor.metrics.values():
            m.reset()

        with tqdm(train_loader, unit="batch") as tepoch:
            for data, domains, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                if (cfg.USE_CUDA):
                    data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                # 2. Representation Learning
                # 2.1. Update the student
                student = representation_learning_batch_training(augmenter, teacher, student, classifier, {"data": data, "target": target, "domains": domains}, m_monitor, logger)
                # 2.2. Update the teacher using Exponential Moving Average (Distillation)
                teacher = ema(teacher, student, cfg.MODEL.TEACHER.TAU)
                # 3. update the augmenter
        if cfg.DEBUG:
            break

def ema(teacher, student, tau):
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data *= tau
            teacher_param.data += (1 - tau) * student_param.data
    return teacher

def warmup(cfg, backbone, classifier, m_monitor, logger):
    phase = 'warmup'

    backbone.unfreeze()
    classifier.unfreeze()
    backbone.train()
    classifier.train()

    # get the parameters of both the backbone and the classifier
    merged_parameters = utils.merge_parameters([backbone, classifier])

    optimizer = optim.SGD(merged_parameters, lr=cfg.MODEL.TEACHER.WARMUP_LR, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    _, train_loader = get_dataset(cfg, phase)

    for epoch in range(cfg.MODEL.TEACHER.WARMUP_EPOCHS):
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, domain, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                if (cfg.USE_CUDA):
                    data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                optimizer.zero_grad()

                # computing the output
                output = classifier(backbone(data))

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
    return backbone, classifier

def representation_learning_batch_training(augmenter, teacher, student, classifier, batch, m_monitor, logger):
    
    # freeze the augmenter and the teacher (we freeze the teacher because we will update it using EMA)
    augmenter.freeze()
    teacher.freeze() # No need to accumulate gradients to make training faster
    # put teacher and student in train mode
    teacher.train()
    student.train()

    optimizer = optim.SGD(student.parameters(), lr=cfg.MODEL.STUDENT.LR)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    # discrepancy_loss is the magnitude of the difference between the teacher and the student normalized outputs
    discripancy_loss = torch.linalg.vector_norm

    # first pass the input through the augmenter and the teacher
    augmented_data = augmenter(batch['data'])
    teacher_output = teacher(batch['data'])
    # then pass the augmented_data through the student
    student_output = student(augmented_data)
    # compute the discrepancy loss between the teacher and the student
    discrepancy = discripancy_loss(teacher_output - student_output, dim=1).sum()
    # compute the cross entropy loss between the student output and the target
    cross_entropy = cross_entropy_loss(classifier(student_output), batch['target'])
    # compute the total loss and update the student
    loss = cross_entropy + discrepancy
    loss.backward()
    optimizer.step()
    # zero the gradients
    optimizer.zero_grad()

    return student

def domain_augmentation_batch_training(augmenter, teacher, student, classifier, batch, m_monitor, logger):
    # unfreeze the augmenter
    augmenter.train()
    augmenter.unfreeze()
    # freeze the teacher and the student
    teacher.train()
    student.train()
    teacher.freeze()
    student.freeze()

    optimizer = optim.SGD(augmenter.parameters(), lr=cfg.MODEL.AUGMENTER.LR, momentum=0.9)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    # discrepancy_loss is the magnitude of the difference between the teacher and the student normalized outputs
    discripancy_loss = torch.linalg.vector_norm

    # compute the centroids of domains in the batch
    with torch.no_grad():
        centroids = []
        for domain in batch['domains']:
            centroids.append(torch.mean(batch['data'][batch['domains'] == domain], dim=0))
        # compute the average distance between the centroids
        margin = torch.mean(torch.linalg.vector_norm(torch.stack(centroids) - torch.stack(centroids).T, dim=1))

    # compute the output of the augmenter
    augmented_data = augmenter(batch['data'])
    # compute the output of the student on the augmented data
    student_output = student(augmented_data)
    # compute the output of the teacher on the original data
    teacher_output = teacher(batch['data'])
    # compute the discrepancy between the teacher and the student
    discrepancy = discripancy_loss(teacher_output - student_output, dim=1)
    # compute the cross entropy loss between the student output and the target
    cross_entropy = cross_entropy_loss(classifier(student_output), batch['target'])
    # compute the minimum between 0 and the margin - discrepancy
    margin_loss = torch.min(discrepancy - margin, torch.tensor(0))
    # compute the total loss and update the augmenter
    loss = -margin_loss + cross_entropy
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return augmenter


if __name__ == '__main__':

    # python train.py --experiment_cfg Experiments/E1_No_Normalization.yaml --use_cuda --use_wandb --experiment_name "testing the warmup process"
    # python train.py --experiment_cfg Experiments/E1_No_Normalization.yaml --use_wandb --experiment_name "testing the warmup process"
    
    args = parser.parse_args()
    cfg = setup_cfg(args)
    # Set the random seed for reproducible experiments
    utils.set_random_seed(cfg.SEED)
    print(cfg)
    logger = utils.set_logger(cfg)
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



