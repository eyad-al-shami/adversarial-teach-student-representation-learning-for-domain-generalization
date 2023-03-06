"""Train the model"""
from commandline.cmdParser import parser
import logging
import os
from configs import setup_cfg
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
import utils
import model.net as net
from data import get_dataset
from sklearn.metrics.pairwise import euclidean_distances

def create_componenets(cfg):
    """
        create the componenets of the learning framework
        1. The augmenter model.
        2. The Teacher model.
        3. The student model.
        4. The classification layer.
    """
    augmenter = net.build_augmenter(cfg=cfg)
    teacher = net.BackBone(cfg=cfg, component='teacher')
    student = net.BackBone(cfg=cfg, component='student')
    classifier = net.ClassifierLayer(cfg=cfg)
    return augmenter, teacher, student, classifier

def training_validation_loop(cfg, logger):
    
    phase = 'training'
    # create the components of the learning framework
    augmenter, teacher, student, classifier = create_componenets(cfg)
    metrics_monitors = {"teacher_warmup_mm": net.Metrics_Monitor(cfg), "teacher_student_update_mm": net.Metrics_Monitor(cfg), "augmenter_discrepancy_mm": net.Metrics_Monitor(cfg), "augmenter_crossentropy_mm": net.Metrics_Monitor(cfg)}
    _, train_loader = get_dataset(cfg, phase, domains=cfg.DATASET.SOURCE_DOMAINS)
    # move modules to GPU if available
    if (cfg.USE_CUDA):
        augmenter = augmenter.to(cfg.DEVICE)
        teacher = teacher.to(cfg.DEVICE)
        student = student.to(cfg.DEVICE)
        classifier = classifier.to(cfg.DEVICE)
        # move the metrics modules to the GPU
        for mm in metrics_monitors.values():
            for m in mm.metrics.values():
                m.to(cfg.DEVICE)
    t_acc_before_warmup = test_teacher(cfg, teacher, classifier)
    # 1. Warmup
    teacher, classifier = teacher_warmup(cfg, teacher, classifier, metrics_monitors["teacher_warmup_mm"], logger)
    t_acc_after_warmup = test_teacher(cfg, teacher, classifier)
    # print the accuracy of the teacher before and after the warmup using print and + and - signs to make it more readable
    print("+"*50)
    print(f"+      Teacher Accuracy Before Warmup: {t_acc_before_warmup}       +")
    print(f"+      Teacher Accuracy After Warmup: {t_acc_after_warmup}         +")
    print("+"*50)
    # 2 and 3 Representation Learning and Distillation
    for epoch in range(cfg.TRAIN.EPOCHS):
        # reset the metrics
        for m in metrics_monitors.values():
            m.reset()
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                if (cfg.USE_CUDA):
                    batch = [input_data.to(cfg.DEVICE) for input_data in batch]
                # 2. Representation Learning
                # 2.1. Update the student
                student, rl_loss = teach_studnet_batch_training(augmenter, teacher, student, classifier, batch, epoch)
                with torch.no_grad():
                    metrics_monitors["teacher_student_update_mm"].metrics["loss"](rl_loss)
                # 2.2. Update the teacher using Exponential Moving Average (Distillation)
                teacher = update_teacher(teacher, student, cfg.MODEL.TEACHER.TAU)
                # 3. update the augmenter
                augmenter, aug_D_loss, aug_Ce_Loss = augmenter_batch_training(augmenter, teacher, student, classifier, batch)
                with torch.no_grad():
                    metrics_monitors["augmenter_discrepancy_mm"].metrics["loss"](aug_D_loss)
                    metrics_monitors["augmenter_crossentropy_mm"].metrics["loss"](aug_Ce_Loss)
                tepoch.set_postfix(rl_loss=rl_loss, aug_D_loss=aug_D_loss)
            if cfg.LOGGING.ENABLED:
                logger.log({"rl_loss":metrics_monitors["teacher_student_update_mm"].metrics["loss"].compute(), "aug_Disc_loss": metrics_monitors["augmenter_discrepancy_mm"].metrics["loss"].compute(), "aug_Ce_loss": metrics_monitors["augmenter_crossentropy_mm"].metrics["loss"].compute() , "epoch": epoch, "phase": phase})
        for m in metrics_monitors.values():
            m.reset()
        # reduce the learning rate after 30 epochs
        if cfg.DRY_RUN:
            break
        teacher_accuracy = test_teacher(cfg, teacher, classifier)
        logger.log({"teacher_acc": teacher_accuracy, "epoch": epoch})
    return teacher, student, augmenter, classifier

@torch.no_grad()
def update_teacher(teacher, student, keep_rate):
    # with torch.no_grad():
    #     for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
    #         teacher_param.data *= tau
    #         teacher_param.data += (1 - tau) * student_param.data
    # return teacher
    student_model_dict = student.state_dict()
    new_teacher_dict = OrderedDict()
    for key, value in teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                (student_model_dict[key] *
                (1 - keep_rate)) + (value * keep_rate)
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    teacher.load_state_dict(new_teacher_dict)
    return teacher

def teacher_warmup(cfg, teacher, classifier, m_monitor, logger):
    '''
        Warmup the teacher and the classifier
        params:
            cfg: the configuration object
            backbone: the teacher model
            classifier: the classifier layer
            m_monitor: the metrics monitor
            logger: the logger object
        return:
            backbone: the teacher model
            classifier: the classifier layer
    '''

    phase = 'warmup'

    teacher.unfreeze()
    classifier.unfreeze()
    teacher.train()
    classifier.train()

    # get the parameters of both the backbone and the classifier
    merged_parameters = utils.merge_parameters([teacher, classifier])

    optimizer = optim.SGD(merged_parameters, lr=cfg.MODEL.TEACHER.WARMUP_LR)
    criterion = torch.nn.CrossEntropyLoss()
    _, train_loader = get_dataset(cfg, phase, domains=cfg.DATASET.SOURCE_DOMAINS)

    for epoch in range(cfg.MODEL.TEACHER.WARMUP_EPOCHS):
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, domain, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                if (cfg.USE_CUDA):
                    data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                optimizer.zero_grad()

                # computing the output
                output = classifier(teacher(data))

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    acc = m_monitor.metrics["acc"](output, target)
                    m_monitor.metrics["loss"](loss)
                tepoch.set_postfix(loss=loss.item(), acc=acc.item())
            if cfg.LOGGING.ENABLED:
                logger.log({"warmup_loss": m_monitor.metrics["loss"].compute(), "warmup_acc": m_monitor.metrics["acc"].compute(), "epoch": epoch, "phase": phase})
        if cfg.DRY_RUN:
            break
    optimizer.zero_grad()
    teacher.zero_grad()
    classifier.zero_grad()
    m_monitor.reset()
    return teacher, classifier

def teach_studnet_batch_training(augmenter, teacher, student, classifier, batch, epoch):
    ''''
        Train the student model
        params:
            augmenter: the augmenter model
            teacher: the teacher model
            student: the student model
            classifier: the classifier layer
            batch: the batch of data
        return:
            student: the student model
            loss: the loss of the student model
    '''

    # freeze the augmenter and the teacher (we freeze the teacher because we will update it using EMA)
    augmenter.freeze()
    teacher.freeze() # No need to accumulate gradients to make training faster
    # put teacher and student in train mode
    teacher.eval()
    student.train()
    optimizer = optim.SGD(student.parameters(), lr=cfg.MODEL.STUDENT.LR)
    if epoch == 30:
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg.MODEL.STUDENT.LR * cfg.MODEL.STUDENT.LR_DECAY

    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    # first pass the input through the augmenter and the teacher
    augmented_data = augmenter(batch[0])
    t_output = teacher(batch[0])
    # then pass the augmented_data through the student
    s_output = student(augmented_data)
    # with torch.no_grad():
    #     t_output_magnitude = torch.linalg.vector_norm(t_output, dim=1, keepdim=True)
    #     s_output_magnitude = torch.linalg.vector_norm(s_output, dim=1, keepdim=True)
    # t_output_normalized = t_output / t_output_magnitude
    # s_output_normalized = s_output / s_output_magnitude

    t_output_normalized = torch.nn.functional.normalize(t_output, dim=1)
    s_output_normalized = torch.nn.functional.normalize(s_output, dim=1)

    discrepancy = t_output_normalized - s_output_normalized
    # discrepancy_loss = torch.einsum('ij, ij -> i', discrepancy, discrepancy).mean()
    # discrepancy_loss = (torch.linalg.norm(discrepancy, dim=1)**2).mean()
    discrepancy_loss = torch.pow(discrepancy, 2).sum(1).mean()
    # compute the cross entropy loss between the student output and the target
    cross_entropy = cross_entropy_loss(classifier(s_output), batch[2])
    # compute the total loss and update the student
    loss = cross_entropy + discrepancy_loss
    loss.backward()
    optimizer.step()
    # zero the gradients
    optimizer.zero_grad()
    student.zero_grad()
    teacher.zero_grad()
    augmenter.zero_grad()
    classifier.zero_grad()
    
    return student, loss.item()

def augmenter_batch_training(augmenter, teacher, student, classifier, batch):
    ''''
        Train the augmenter model
        params:
            augmenter: the augmenter model
            teacher: the teacher model
            student: the student model
            classifier: the classifier layer
            batch: the batch of data
        return:
            augmenter: the augmenter model
            loss: the loss of the augmenter model
    '''
    
    # unfreeze the augmenter
    augmenter.train()
    augmenter.unfreeze()
    teacher.eval()
    student.eval()

    optimizer = optim.SGD(augmenter.parameters(), lr=cfg.MODEL.AUGMENTER.LR)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    # compute the centroids of domains in the batch
    with torch.no_grad():
        centroids = []
        with torch.no_grad():
            doamins = set(batch[1].tolist())
            for domain in doamins:
                centroids.append(torch.mean(batch[0][batch[1] == domain], dim=[0, 2, 3])) # the output of this line is a tensor of shape (3)
            centroids = torch.vstack(centroids)
            distances = euclidean_distances(centroids.cpu().numpy(), centroids.cpu().numpy())
            distances = distances[np.triu_indices(distances.shape[0], k = 1)]
            margin = np.mean(distances)
    
    margin = torch.tensor(margin).to(cfg.DEVICE)

    augmented_data = augmenter(batch[0])
    s_output = student(augmented_data)
    t_output = teacher(batch[0])

    # with torch.no_grad():
    #     t_output_magnitude = torch.linalg.vector_norm(t_output, dim=1, keepdim=True)
    #     s_output_magnitude = torch.linalg.vector_norm(s_output, dim=1, keepdim=True)
        
    # t_output_normalized = t_output / t_output_magnitude
    # s_output_normalized = s_output / s_output_magnitude

    t_output_normalized = torch.nn.functional.normalize(t_output, dim=1)
    s_output_normalized = torch.nn.functional.normalize(s_output, dim=1)

    discrepancy = t_output_normalized - s_output_normalized
    # discrepancy_loss = torch.einsum('ij, ij -> i', discrepancy, discrepancy)
    # discrepancy_loss = (torch.linalg.norm(discrepancy, dim=1)**2)
    discrepancy_loss = torch.pow(discrepancy, 2).sum(1)
    margin_loss = torch.min(discrepancy_loss - margin, torch.tensor(0)).sum()
    cross_entropy = cross_entropy_loss(classifier(s_output), batch[2])
    
    loss = - margin_loss + cross_entropy 
    loss.backward()
    optimizer.step()
    # zero the gradients
    optimizer.zero_grad()
    student.zero_grad()
    teacher.zero_grad()
    augmenter.zero_grad()
    classifier.zero_grad()
    return augmenter, margin_loss.item(), cross_entropy.item()

def test_teacher(cfg, teacher, classifier):
    _, target_data_loader = get_dataset(cfg, domains=cfg.DATASET.TARGET_DOMAINS)
    teacher.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in target_data_loader:
            if (cfg.USE_CUDA):
                batch = [tensor.to(cfg.DEVICE) for tensor in batch]
            output = teacher(batch[0])
            output = classifier(output)
            _, predicted = torch.max(output.data, 1)
            total += batch[2].size(0)
            correct += (predicted == batch[2]).sum().item()
    return 100 * correct / total

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = setup_cfg(args)

    utils.set_random_seed(cfg.SEED)
    print(cfg)
    logger = utils.set_logger(cfg)
    teacher, student, augmenter, classifier = training_validation_loop(cfg, logger)
    teacher_acc = test_teacher(cfg, teacher, classifier)

    print("The teacher accuracy on the target domain is: ", teacher_acc)

    utils.save_checkpoint(teacher.state_dict(), False, cfg.OUTPUT_DIR)
    utils.save_checkpoint(student.state_dict(), False, cfg.OUTPUT_DIR)
    utils.save_checkpoint(augmenter.state_dict(), False, cfg.OUTPUT_DIR)
    utils.save_checkpoint(classifier.state_dict(), False, cfg.OUTPUT_DIR)



