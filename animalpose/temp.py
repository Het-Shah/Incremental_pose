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

# df1 = pd.read_csv("/media/gaurav/Incremental_pose/data/updated_df_rotated_random_all_three_cat.csv")
# df2 = pd.read_csv("/media/gaurav/Incremental_pose/data/updated_df_rotated_random_all_three_dog.csv")
# df3 = pd.read_csv("/media/gaurav/Incremental_pose/data/updated_df_rotated_random_all_three_cow.csv")
# df4 = pd.read_csv("/media/gaurav/Incremental_pose/data/updated_df_rotated_random_all_three_horse.csv")
# df5 = pd.read_csv("/media/gaurav/Incremental_pose/data/updated_df_rotated_random_all_three_sheep.csv")

# df = pd.concat([df1, df2, df3, df4, df5])
# df.to_csv("/media/gaurav/Incremental_pose/data/updated_df_rotated_random_all_three.csv", index=False)

rotate_elbow_knee_limbs(
    input_dir="/media/gaurav/Incremental_pose/data/cropped_images_no_zero_padding",
    output_dir="/media/gaurav/Incremental_pose/data/rotated_images_no_zero_padding/",
    input_csv_file="/media/gaurav/Incremental_pose/data/updated_df_no_zero_padding.csv",
    output_csv_file="/media/gaurav/Incremental_pose/data/updated_df_rotated_no_zero_padding.csv",
    animal_class="cat",
)