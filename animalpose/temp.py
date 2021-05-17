# """Script for multi-gpu training for incremental learing."""
import json
import os
import shutil
import random
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from PIL import Image
import PIL.ImageDraw as ImageDraw

from dppy.finite_dpps import FiniteDPP

import torch
import torch.nn as nn
import torchvision
import torch.utils.data
import torchvision.models as models

from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchsummary import summary

from alphapose.models import builder

# from alphapose.opt import cfg, logger, opt
from alphapose.utils.logger import board_writing, debug_writing
from alphapose.utils.metrics import DataLogger, calc_accuracy
from alphapose.utils.transforms import get_max_pred_batch
from transforms_utils import get_max_pred
from animal_data_loader import AnimalDatasetCombined, ToTensor

import thinplate as tps
from ray.util.multiprocessing import Pool

from utils import *

# df1 = pd.read_csv("/media2/het/Incremental_pose/data/updated_df_rotated_random_all_three_cat.csv")
# df2 = pd.read_csv("/media2/het/Incremental_pose/data/updated_df_rotated_random_all_three_dog.csv")
# df3 = pd.read_csv("/media2/het/Incremental_pose/data/updated_df_rotated_random_all_three_cow.csv")
# df4 = pd.read_csv("/media2/het/Incremental_pose/data/updated_df_rotated_random_all_three_horse.csv")
# df5 = pd.read_csv("/media2/het/Incremental_pose/data/updated_df_rotated_random_all_three_sheep.csv")

# df = pd.concat([df1, df2, df3, df4, df5])
# df.to_csv("/media2/het/Incremental_pose/data/updated_df_rotated_random_all_three.csv", index=False)
# print(df.head())

rotate_elbow_knee_limbs(
    input_dir="/media2/het/Incremental_pose/data/cropped_images",
    output_dir="/media2/het/Incremental_pose/data/temp/",
    input_csv_file="/media2/het/Incremental_pose/data/updated_df.csv",
    output_csv_file="/media2/het/Incremental_pose/data/updated_df_rotated_no_zero_padding_1.csv",
    animal_class="cow",
)


# """Script for multi-gpu training for incremental learing."""
# import json
# import os
# from copy import deepcopy

# import scipy.linalg as la

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

# from dppy.finite_dpps import FiniteDPP
# from dppy.utils import example_eval_L_linear

# import torch
# import torch.nn as nn
# import torchvision
# import torch.utils.data
# from tensorboardX import SummaryWriter
# from tqdm import tqdm

# from alphapose.models import builder
# from alphapose.opt import cfg, logger, opt
# from alphapose.utils.logger import board_writing, debug_writing
# from alphapose.utils.metrics import DataLogger, calc_accuracy, get_max_pred_batch
# from transforms_utils import get_max_pred
# from animal_data_loader import AnimalDatasetCombined, ToTensor

# from utils import *
# from models import *

# num_gpu = torch.cuda.device_count()
# valid_batch = 1 * num_gpu
# if opt.sync:
#     norm_layer = nn.SyncBatchNorm
# else:
#     norm_layer = nn.BatchNorm2d

# def validate(m, val_loader, opt, cfg, writer, criterion, batch_size=1):
#     loss_logger_val = DataLogger()
#     acc_logger = DataLogger()

#     m.eval()

#     # val_loader = tqdm(val_loader, dynamic_ncols=True)

#     cnt = 0

#     for inps, labels, label_masks, _ in val_loader:
#         if isinstance(inps, list):
#             inps = [inp.cuda() for inp in inps]

#         else:
#             inps = inps.cuda()
#         labels = labels.cuda()
#         label_masks = label_masks.cuda()

#         output = m(inps)

#         final_img = np.zeros((128, 128))
#         final_img_pred = np.zeros((128, 128))

#         input_img = inps.cpu().numpy().transpose((0, 2, 3, 1))
#         input_img = np.squeeze(input_img, axis=0)

#         input_img = Image.fromarray(input_img.astype(np.uint8))
#         input_img.save(
#             os.path.join(
#                 "./temp/", "image" + "_{}.png".format(cnt)
#             )
#         )

#         for k in range(labels.shape[1]):
#             images = labels.cpu().numpy()[0][k]
#             final_img += images

#         plt.imsave(
#             os.path.join(
#                 "./temp/", "temp" + "_{}.png".format(cnt)
#             ),
#             final_img,
#             cmap="gray",
#         )

#         for k in range(output.shape[1]):
#             images = output.cpu().numpy()[0][k]
#             final_img_pred += images

#         plt.imsave(
#             os.path.join(
#                 "./temp/", "pred" + "_{}.png".format(cnt)
#             ),
#             final_img_pred,
#             cmap="gray",
#         )

#         acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))

#         preds, _ = get_max_pred_batch(output.mul(label_masks).detach().cpu().numpy())
#         labels, _ = get_max_pred_batch(labels.mul(label_masks).detach().cpu().numpy())

#         # print(preds)

#         # print(labels)

#         # print(acc)

#         acc_logger.update(acc, batch_size)

#         # TQDM
#         # val_loader.set_description(
#         #     "Loss: {loss:.4f} acc: {acc:.4f}".format(
#         #         loss=loss_logger_val.avg, acc=acc_logger.avg
#         #     )
#         # )
#         if cnt == 10:
#             break

#         cnt+=1

#     # val_loader.close()
#     return loss_logger_val.avg, acc_logger.avg

# def main():
#     # List of keypoints used
#     # 0-2, 1-2, 0-3, 1-3, 5-13, 13-6, 8-14, 14-7, 11-15, 15-9, 12-16, 16-10  => 12 limbs

#     keypoint_names = [
#         "L_Eye",
#         "R_Eye",
#         "Nose",
#         "L_EarBase",
#         "R_EarBase",
#         "L_F_Elbow",
#         "L_F_Paw",
#         "R_F_Paw",
#         "R_F_Elbow",
#         "L_B_Paw",
#         "R_B_Paw",
#         "L_B_Elbow",
#         "R_B_Elbow",
#         "L_F_Knee",
#         "R_F_Knee",
#         "L_B_Knee",
#         "R_B_Knee",
#     ]

#     # Model Initialize
#     m = preset_model(cfg)
#     m = nn.DataParallel(m).cuda()

#     if cfg.MODEL.PRETRAINED:
#         logger.info(f"Loading model from {cfg.MODEL.PRETRAINED}...")
#         m.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))

#     if len(cfg.ANIMAL_CLASS_INCREMENTAL) % cfg.INCREMENTAL_STEP != 0:
#         print(
#             "Number of classes for incremental step is not a multiple of the number of incremental steps!"
#         )
#         return

#     if cfg.LOSS.TYPE == "SmoothL1":
#         criterion = nn.SmoothL1Loss().cuda()
#     else:
#         criterion = builder.build_loss(cfg.LOSS).cuda()

#     writer = SummaryWriter(".tensorboard/{}-{}".format(opt.exp_id, cfg.FILE_NAME))

#     # generating base data loaders
#     annot_df = pd.read_csv(cfg.DATASET.ANNOT)

#     train_datasets = []
#     val_datasets = []

#     classes_till_now = []

#     filename_list_classes = {}

#     for animal_class in cfg.ANIMAL_CLASS_BASE:
#         classes_till_now.append(animal_class)
#         temp_df = annot_df.loc[annot_df["class"] == animal_class]

#         images_list = np.array(temp_df["filename"])
#         np.random.seed(121)
#         np.random.shuffle(images_list)

#         val_images_list = images_list[int(0.9 * len(images_list)) :]

#         val_tempset = AnimalDatasetCombined(
#             cfg.DATASET.IMAGES,
#             cfg.DATASET.ANNOT,
#             val_images_list,
#             input_size=(512, 512),
#             output_size=(128, 128),
#             transforms=torchvision.transforms.Compose([ToTensor()]),
#             train=False,
#         )

#         val_datasets.append(val_tempset)

#     base_valset = torch.utils.data.ConcatDataset(val_datasets)
#     base_val_loader = torch.utils.data.DataLoader(
#         base_valset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE
#     )

#     # Prediction Test
#     with torch.no_grad():
#         val_loss, val_acc = validate(
#             m,
#             base_val_loader,
#             opt,
#             cfg,
#             writer,
#             criterion,
#             batch_size=cfg.TRAIN.VAL_BATCH_SIZE,
#         )


# def preset_model(cfg):
#     if cfg.MODEL.TYPE == "custom":
#         model = DeepLabCut()
#     else:
#         model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

#     logger.info("Create new model")
#     logger.info("=> init weights")
#     if cfg.MODEL.TYPE != "custom":
#         model._initialize()

#     return model


# if __name__ == "__main__":
#     main()
