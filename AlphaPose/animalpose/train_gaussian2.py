"""Script for multi-gpu training for incremental learing."""
import json
import os
from copy import deepcopy

import scipy.linalg as la

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data
from tensorboardX import SummaryWriter
from torchsummary import summary

from tqdm import tqdm

from alphapose.models import builder
from alphapose.opt import cfg, logger, opt
from alphapose.utils.logger import board_writing, debug_writing
from alphapose.utils.metrics import DataLogger, calc_accuracy
from alphapose.utils.transforms import get_max_pred_batch
from transforms_utils import get_max_pred
from animal_data_loader import AnimalDatasetCombined, ToTensor

from utils import *
from models import *

num_gpu = torch.cuda.device_count()
valid_batch = 1 * num_gpu
if opt.sync:
    norm_layer = nn.SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


class gaussian_model(nn.Module):
    def __init__(self):
        super(gaussian_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, (3, 3))
        self.conv2 = nn.Conv2d(128, 256, (3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 30 * 30, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 13)

    def forward(self, gauss_maps):
        out2 = self.pool(F.relu(self.conv1(gauss_maps)))
        out2 = self.pool(F.relu(self.conv2(out2)))
        out2 = out2.view(-1, 256 * 30 * 30)
        out2 = F.relu(self.fc1(out2))
        out2 = F.relu(self.fc2(out2))
        out2 = self.fc3(out2)
        return out2


def train(
    opt,
    train_loader,
    m,
    gauss_model,
    criterion_MSE,
    criterion_CrossEntropy,
    optimizer,
    writer,
    phase="Train",
):
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    class_acc_logger = DataLogger()
    if cfg.C1.TRAIN:
        m.train()
    else:
        m.eval()

    gauss_model.train()

    train_loader = tqdm(train_loader, dynamic_ncols=True)

    total = 0
    correct = 0

    for i, (inps, labels, label_masks, _) in enumerate(train_loader):
        if isinstance(inps, list):
            inps = [inp.cuda().requires_grad_() for inp in inps]
        else:
            inps = inps.cuda().requires_grad_()
        labels = labels.cuda()
        label_masks = label_masks.cuda()

        output = m(inps)

        optimizer.zero_grad()
        if cfg.C1.TRAIN:
            if cfg.LOSS.TYPE == "SmoothL1":
                loss_mse = criterion(
                    output.mul(label_masks), labels.mul(label_masks)
                ).item()

            if cfg.LOSS.get("TYPE") == "MSELoss":
                loss_mse = (
                    0.5
                    * criterion_MSE(
                        output.mul(label_masks), labels.mul(label_masks)
                    ).item()
                )

            # loss_mse.backward()

        acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        if cfg.C2.TRAIN:
            joints, vis = get_max_pred_batch(labels.mul(label_masks).cpu().data.numpy())

            gauss_maps, labels = limb_maps(joints, vis)
            gauss_maps = gauss_maps.cuda()
            labels = labels.long().cuda()

            loss_ce = 0
            for joints in range(12):
                inp = torch.unsqueeze(gauss_maps[:, joints], 1)
                output_2 = gauss_model(inp)
                try:
                    loss_ce += criterion_CrossEntropy(output_2, labels[:, joints])
                    _, predicted = torch.max(output_2, 1)
                    correct += (predicted == labels[:, joints]).sum().item()
                    total += labels[:, joints].size(0)
                except:
                    continue

        # class_acc = correct/total
        if cfg.C1.TRAIN and cfg.C2.TRAIN:
            loss = loss_mse + 0.5 * loss_ce
        elif cfg.C2.TRAIN:
            loss = loss_ce
        elif cfg.C1.TRAIN:
            loss = loss_mse

        try:
            loss_logger.update(loss.item(), batch_size)
        except:
            loss_logger.update(loss, batch_size)
        acc_logger.update(acc, batch_size)
        if cfg.C2.TRAIN:
            class_acc_logger.update(correct / total, 1)

        try:
            loss.backward()
        except:
            pass
        optimizer.step()

        opt.trainIters += 1
        # Tensorboard
        if opt.board:
            board_writing(
                writer, loss_logger.avg, acc_logger.avg, opt.trainIters, phase
            )

        # Debug
        if opt.debug and not i % 10:
            debug_writing(writer, output, labels, inps, opt.trainIters)

        # TQDM
        if cfg.C1.TRAIN and cfg.C2.TRAIN:
            train_loader.set_description(
                "loss: {loss:.8f} | acc: {acc:.4f} | acc_classifier:  {class_acc:.4f}".format(
                    loss=loss_logger.avg, acc=acc_logger.avg, class_acc=correct / total
                )
            )
        elif cfg.C1.TRAIN:
            train_loader.set_description(
                "loss: {loss:.8f} | acc: {acc:.4f} ".format(
                    loss=loss_logger.avg, acc=acc_logger.avg
                )
            )
        elif cfg.C2.TRAIN:
            train_loader.set_description(
                "loss: {loss:.8f} | acc: {acc:.4f} | acc_classifier:  {class_acc:.4f}".format(
                    loss=loss_logger.avg, acc=acc_logger.avg, class_acc=correct / total
                )
            )

    train_loader.close()

    return loss_logger.avg, acc_logger.avg


def validate(m, val_loader, opt, cfg, writer, criterion, batch_size=1):
    loss_logger_val = DataLogger()
    acc_logger = DataLogger()

    m.eval()

    val_loader = tqdm(val_loader, dynamic_ncols=True)

    for inps, labels, label_masks, _ in val_loader:
        if isinstance(inps, list):
            inps = [inp.cuda() for inp in inps]

        else:
            inps = inps.cuda()
        labels = labels.cuda()
        label_masks = label_masks.cuda()

        output = m(inps)

        loss = criterion(output.mul(label_masks), labels.mul(label_masks))
        acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))

        loss_logger_val.update(loss, batch_size)
        acc_logger.update(acc, batch_size)

        # TQDM
        val_loader.set_description(
            "Loss: {loss:.4f} acc: {acc:.4f}".format(
                loss=loss_logger_val.avg, acc=acc_logger.avg
            )
        )

    val_loader.close()
    return loss_logger_val.avg, acc_logger.avg


def preset_model(cfg):
    if cfg.MODEL.TYPE == "custom":
        model = DeepLabCut()
    else:
        model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    if cfg.MODEL.TYPE != "custom":
        logger.info("Create new model")
        logger.info("=> init weights")
        model._initialize()

    return model


def main():
    logger.info("******************************")
    logger.info(opt)
    logger.info("******************************")
    logger.info(cfg)
    logger.info("******************************")

    keypoint_names = [
        "L_Eye",
        "R_Eye",
        "Nose",
        "L_EarBase",
        "R_EarBase",
        "L_F_Elbow",
        "L_F_Paw",
        "R_F_Paw",
        "R_F_Elbow",
        "L_B_Paw",
        "R_B_Paw",
        "L_B_Elbow",
        "R_B_Elbow",
        "L_F_Knee",
        "R_F_Knee",
        "L_B_Knee",
        "R_B_Knee",
    ]

    # Models Initialize
    m = preset_model(cfg).cuda()

    gauss_model = gaussian_model().cuda()
    gauss_model.apply(init_weights)

    if cfg.C1.PRETRAINED:
        logger.info(f"Loading model from {cfg.C1.PRETRAINED}...")
        m.load_state_dict(torch.load(cfg.C1.PRETRAINED))

    if cfg.C2.PRETRAINED:
        logger.info(f"Loading model from {cfg.C2.PRETRAINED}...")
        gauss_model.load_state_dict(torch.load(cfg.C2.PRETRAINED))

    if cfg.TRAIN.OPTIMIZER == "adam":
        if cfg.C1.TRAIN and cfg.C2.TRAIN:
            optimizer = torch.optim.Adam(
                list(m.parameters()) + list(gauss_model.parameters()),
                lr=cfg.TRAIN.LR,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            )
        elif cfg.C1.TRAIN:
            optimizer = torch.optim.Adam(
                m.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            )
        elif cfg.C2.TRAIN:
            optimizer = torch.optim.Adam(
                gauss_model.parameters(),
                lr=cfg.TRAIN.LR,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            )

    criterion_MSE = builder.build_loss(cfg.LOSS).cuda()
    criterion_CrossEntropy = nn.CrossEntropyLoss().cuda()

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR
    )

    writer = SummaryWriter(".tensorboard/{}-{}".format(opt.exp_id, cfg.FILE_NAME))

    annot_df = pd.read_csv(cfg.DATASET.ANNOT)

    train_datasets = []
    val_datasets = []

    classes_till_now = []

    filename_list_classes = {}

    for animal_class in cfg.ANIMAL_CLASS_BASE:
        classes_till_now.append(animal_class)
        temp_df = annot_df.loc[annot_df["class"] == animal_class]

        images_list = np.array(temp_df["filename"])
        np.random.seed(121)
        np.random.shuffle(images_list)

        train_images_list = images_list[: int(0.9 * len(images_list))]
        val_images_list = images_list[int(0.9 * len(images_list)) :]

        train_tempset = AnimalDatasetCombined(
            cfg.DATASET.IMAGES,
            cfg.DATASET.ANNOT,
            train_images_list,
            input_size=(512, 512),
            output_size=(128, 128),
            transforms=torchvision.transforms.Compose([ToTensor()]),
            train=True,
        )

        val_tempset = AnimalDatasetCombined(
            cfg.DATASET.IMAGES,
            cfg.DATASET.ANNOT,
            val_images_list,
            input_size=(512, 512),
            output_size=(128, 128),
            transforms=torchvision.transforms.Compose([ToTensor()]),
            train=False,
        )

        train_datasets.append(train_tempset)
        val_datasets.append(val_tempset)

        filename_list_classes[animal_class] = []
        for i in train_images_list:
            filename_list_classes[animal_class].append(i)

    base_trainset = torch.utils.data.ConcatDataset(train_datasets)
    base_train_loader = torch.utils.data.DataLoader(
        base_trainset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True
    )

    base_valset = torch.utils.data.ConcatDataset(val_datasets)
    base_val_loader = torch.utils.data.DataLoader(
        base_valset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE
    )

    opt.trainIters = 0
    opt.val_iters = 0

    best_acc = 0.0
    best_model_weights = deepcopy(m.state_dict())
    best_c2_model_weights = deepcopy(gauss_model.state_dict())

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]

        logger.info(
            f"############# Starting Epoch {opt.epoch} | LR: {current_lr} #############"
        )

        # Training

        train_loss, train_acc = train(
            opt,
            base_train_loader,
            m,
            gauss_model,
            criterion_MSE,
            criterion_CrossEntropy,
            optimizer,
            writer,
            phase="Train",
        )
        logger.epochInfo("Train", opt.epoch, train_loss, train_acc)

        lr_scheduler.step()

        # Prediction Test
        with torch.no_grad():
            val_loss, val_acc = validate(
                m,
                base_val_loader,
                opt,
                cfg,
                writer,
                criterion_MSE,
                batch_size=cfg.TRAIN.VAL_BATCH_SIZE,
            )
            logger.info(
                f"##### Epoch {opt.epoch} | Loss: {val_loss} | acc: {val_acc} #####"
            )

            # Tensorboard
            if opt.board:
                board_writing(writer, val_loss, val_acc, opt.val_iters, "Val")

            opt.val_iters += 1

        # if val_acc > best_acc:
        # Save best weights
        best_model_weights = deepcopy(m.state_dict())
        best_c2_model_weights = deepcopy(gauss_model.state_dict())
        # best_acc = val_acc

        # Time to add DPG
        if i == cfg.TRAIN.DPG_MILESTONE:
            torch.save(
                best_model_weights,
                "./exp/{}-{}/model_{}.pth".format(opt.exp_id, cfg.FILE_NAME, "Base"),
            )
            # Adjust learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.TRAIN.LR

            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=cfg.TRAIN.DPG_STEP, gamma=0.1
            )

            base_trainset = torch.utils.data.ConcatDataset(train_datasets)
            base_train_loader = torch.utils.data.DataLoader(
                base_trainset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True
            )

    if cfg.C1.TRAIN:
        torch.save(
            best_model_weights,
            "./exp/{}-{}/final_weights.pth".format(opt.exp_id, cfg.FILE_NAME),
        )

    if cfg.C2.TRAIN:
        torch.save(
            best_c2_model_weights,
            "./exp/{}-{}/final_weights_c2.pth".format(opt.exp_id, cfg.FILE_NAME),
        )


if __name__ == "__main__":
    main()
