# """Script for multi-gpu training for incremental learing."""
import json
import os
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

from utils import *

# List of keypoints used
# keypoint_names = [
#     "L_Eye",
#     "R_Eye",
#     "Nose",
#     "L_EarBase",
#     "R_EarBase",
#     "L_F_Elbow",
#     "L_F_Paw",
#     "R_F_Paw",
#     "R_F_Elbow",
#     "L_B_Paw",
#     "R_B_Paw",
#     "L_B_Elbow",
#     "R_B_Elbow",
#     "L_F_Knee",
#     "R_F_Knee",
#     "L_B_Knee",
#     "R_B_Knee",
# ]

# filename_list_classes = {}

# annot_df = pd.read_csv("/media2/het/Incremental_pose/data/updated_df.csv")

# print(annot_df.loc[annot_df["class"] == "horse"])
# temp_df = annot_df.loc[annot_df["class"] == "dog"]
# images_list = np.array(temp_df["filename"])
# np.random.seed(421)
# np.random.shuffle(images_list)
# images_list = images_list[:5]

# animal_class = "cat"

# trainset = AnimalDatasetCombined(
#     "/media2/het/Incremental_pose/data/cropped_images/",
#     "/media2/het/Incremental_pose/data/updated_df.csv",
#     images_list,
#     input_size=(512, 512),
#     output_size=(128, 128),
#     transforms=torchvision.transforms.Compose([ToTensor()]),
#     train=True,
#     parts_augmentation=True,
# )

# train_loader = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=False)


# def get_rot_gaussian_maps(mu, shape_hw, inv_std1, inv_std2, angles, mode="rot"):
#     """
# 	Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
# 	given the gaussian centers: MU [B, NMAPS, 2] tensor.

# 	STD: is the fixed standard dev.
# 	"""
#     mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]  # (B, 12, 1)

#     y = np.linspace(-1.0, 1.0, shape_hw[0])

#     x = np.linspace(-1.0, 1.0, shape_hw[1])  # Bx14

#     y = np.reshape(np.tile(y, [shape_hw[0]]), (-1, shape_hw[0], shape_hw[0]))
#     y = np.expand_dims(y, 0) * np.ones((mu.shape[1], shape_hw[0], shape_hw[0]))

#     x = np.reshape(
#         np.tile(x, [shape_hw[1]]), (-1, shape_hw[1], shape_hw[1])
#     )  # Bx128x128
#     x = np.expand_dims(x, 0) * np.ones(
#         (mu.shape[1], shape_hw[1], shape_hw[1])
#     )  # Bx12x128x128
#     x = np.transpose(x, [0, 1, 3, 2])
#     mu_y, mu_x = np.expand_dims(mu_y, 3), np.expand_dims(mu_x, 3)  # Bx12x1x1

#     y = y - mu_y
#     x = x - mu_x  # Bx12x128x128

#     if mode in ["rot", "flat"]:
#         # apply rotation to the grid
#         yx_stacked = np.stack(
#             [
#                 np.reshape(y, (-1, y.shape[1], y.shape[2] * y.shape[3])),
#                 np.reshape(x, (-1, x.shape[1], x.shape[2] * x.shape[3])),
#             ],
#             2,
#         )  # (B, 12, 2, 128^2)
#         rot_mat = np.stack(
#             [
#                 np.stack([np.cos(angles), np.sin(angles)], 2),
#                 np.stack([-np.sin(angles), np.cos(angles)], 2),
#             ],
#             3,
#         )  # (B, 128, 2, 2)

#         rotated = np.matmul(rot_mat, yx_stacked)  # (B, 12, 2, 128^2)

#         y_rot = rotated[:, :, 0, :]  # (B, 12, 128^2)
#         x_rot = rotated[:, :, 1, :]  # (B, 12, 128^2)

#         y_rot = np.reshape(
#             y_rot, (-1, mu.shape[1], shape_hw[0], shape_hw[0])
#         )  # (B, 12, 128, 128)
#         x_rot = np.reshape(
#             x_rot, (-1, mu.shape[1], shape_hw[1], shape_hw[1])
#         )  # (B, 12, 128, 128)

#         g_y = np.square(y_rot)  # (B, 12, 128, 128)
#         g_x = np.square(x_rot)  # (B, 12, 128, 128)

#         inv_std1 = np.expand_dims(np.expand_dims(inv_std1, 2), 2)  # Bx12x1x1
#         inv_std2 = np.expand_dims(np.expand_dims(inv_std2, 2), 2)  # Bx12x1x1
#         dist = g_y * inv_std1 ** 2 + g_x * inv_std2 ** 2

#         if mode == "rot":
#             g_yx = np.exp(-dist)

#         else:
#             g_yx = np.exp(-np.pow(dist + 1e-5, 0.25))

#     else:
#         raise ValueError("Unknown mode: " + str(mode))

#     g_yx = np.transpose(g_yx, [0, 1, 3, 2])
#     g_yx = torch.FloatTensor(g_yx)

#     return g_yx


# def get_limb_centers(joints_2d, vis):
#     limb_parents_dict = {
#         0: [3],
#         1: [4],
#         2: [0, 1],
#         13: [5],
#         6: [13],
#         14: [8],
#         7: [14],
#         15: [11],
#         9: [15],
#         16: [12],
#         10: [16],
#     }
#     angles_x = []
#     angles_y = []
#     limbs_x = []
#     limbs_y = []
#     limb_length = []
#     labels = []

#     cnt = 0
#     for i in limb_parents_dict.keys():
#         for j in limb_parents_dict[i]:
#             vis_pair = np.squeeze(
#                 np.stack(
#                     [
#                         np.logical_and(vis[:, i], vis[:, j]).astype(int),
#                         np.logical_and(vis[:, i], vis[:, j]).astype(int),
#                     ]
#                 )
#             )

#             temp_label = []

#             for v in np.logical_and(vis[:, i], vis[:, j]):
#                 for t in v:
#                     if t == 0:
#                         temp_label.append(12)
#                     else:
#                         temp_label.append(cnt)

#             labels.append(temp_label)
#             cnt += 1
#             x_pair = np.array([joints_2d[:, i, 0], joints_2d[:, j, 0]])
#             x_pair = np.multiply(
#                 x_pair, vis_pair
#             )  # Do not take limbs where keypoints are not visible

#             y_pair = [joints_2d[:, i, 1], joints_2d[:, j, 1]]
#             y_pair = np.multiply(y_pair, vis_pair)

#             limbs_x.append((x_pair[0] + x_pair[1]) / 2.0)
#             limbs_y.append((y_pair[0] + y_pair[1]) / 2.0)
#             limb_length.append(
#                 np.sqrt((x_pair[0] - x_pair[1]) ** 2 + (y_pair[0] - y_pair[1]) ** 2)
#             )

#             angles_x.append(x_pair[1] - x_pair[0])  # because y is represented as x
#             angles_y.append(y_pair[0] - y_pair[1])

#     angles_x = np.stack(angles_x, 1)
#     angles_y = np.stack(angles_y, 1)

#     angles = np.arctan2(angles_x, angles_y + 1e-7)  # x/y as pose is passed as (y,x)

#     limbs_x = np.stack(limbs_x, 1)
#     limbs_y = np.stack(limbs_y, 1)

#     limbs = np.stack([limbs_x, limbs_y], 2)
#     limb_length = np.stack(limb_length, 1)

#     labels = np.array(labels).T
#     return limbs, angles, limb_length, labels


# def limb_maps(pose_points, vis):
#     points_exchanged = (
#         np.stack([pose_points[:, :, 1], pose_points[:, :, 0]], 2) / 64.0 - 1.0
#     )

#     limb_centers_yx, angles, limb_length, labels = get_limb_centers(
#         points_exchanged, vis
#     )
#     # decreasing the value of ratio increases the length of the gaussian
#     length_ratios = []
#     for i in limb_length:
#         temp = []
#         for j in i:
#             if j:
#                 temp.append(3 / j)
#             else:
#                 temp.append(0)
#         length_ratios.append(temp)

#     length_ratios = np.array(length_ratios)
#     # decreasing the value of ratio increases the width of the gaussian
#     width_ratios = np.array(
#         [3.0, 3.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
#     ) * (np.ones_like(limb_length))

#     gauss_map = get_rot_gaussian_maps(
#         limb_centers_yx, [128, 128], width_ratios, length_ratios, angles, mode="rot"
#     )

#     labels = torch.FloatTensor(labels)

#     return gauss_map, labels


# # (img, label, label_mask, _) = next(iter(train_loader))

# # for img, label, label_mask, idx in train_loader:
# # 	label_mask = torch.squeeze(label_mask)
# # 	ones = torch.ones_like(label_mask)
# # 	if torch.all(label_mask == ones).item():
# # 		print(idx)

# # for i in range(img.shape[0]):
# # 	inp = img[i].cpu().data.numpy().transpose((1,2,0))

# # 	inp = Image.fromarray(inp.astype(np.uint8))
# # 	inp.save("temp"+str(i)+".png")

# # print(label.shape)

# # joints, vis = get_max_pred_batch(label.mul(label_mask).cpu().data.numpy())

# # gauss_map, labels = limb_maps(joints, vis)

# # print(labels.shape)
# # print(labels)

# # {0: [3], 1: [4], 2: [0, 1], 13: [5], 6: [13], 14: [8], 7: [14], 15: [11], 9: [15], 16: [12], 10: [16]}

# # for i in range(12):
# # 	plt.imsave("./temp/temp"+str(i)+".png", gauss_map[0][i])

# # label = label.cpu().numpy()

# # label_sum = label[0][0]
# # for i in range(1,17):
# # 	label_sum += label[0][i]

# # plt.imsave("temp.png",label_sum)


def get_keypoints(fname):
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

    keypoints = []
    vis = []

    annot_df = pd.read_csv("/media2/het/Incremental_pose/data/updated_df.csv")

    temp = annot_df.loc[annot_df["filename"] == fname]
    for keypt in keypoint_names:
        keypoints.append((temp[keypt + "_x"].item(), temp[keypt + "_y"].item()))
        vis.append(temp[keypt + "_visible"].item())

    keypoints_dict = {}
    keypoints_dict["L_Eye"] = keypoints[0]
    keypoints_dict["R_Eye"] = keypoints[1]
    keypoints_dict["Nose"] = keypoints[2]
    keypoints_dict["L_Ear"] = keypoints[3]
    keypoints_dict["R_Ear"] = keypoints[4]
    keypoints_dict["LF_Elbow"] = keypoints[5]
    keypoints_dict["LF_Paw"] = keypoints[6]
    keypoints_dict["RF_Paw"] = keypoints[7]
    keypoints_dict["RF_Elbow"] = keypoints[8]
    keypoints_dict["LB_Paw"] = keypoints[9]
    keypoints_dict["RB_Paw"] = keypoints[10]
    keypoints_dict["LB_Elbow"] = keypoints[11]
    keypoints_dict["RB_Elbow"] = keypoints[12]
    keypoints_dict["LF_Knee"] = keypoints[13]
    keypoints_dict["RF_Knee"] = keypoints[14]
    keypoints_dict["LB_Knee"] = keypoints[15]
    keypoints_dict["RB_Knee"] = keypoints[16]
    return keypoints, keypoints_dict, vis


def get_xs_ys(annot_list):
    x, y = [], []
    for i in range(len(annot_list)):
        if i % 2 == 0:
            x.append(annot_list[i])
        else:
            y.append(annot_list[i])

    x.append(x[0])
    y.append(y[0])
    return x, y


def rotate_about_pt(x, y, origin_x, origin_y, angle):
    x_ = x - origin_x
    y_ = y - origin_y
    c = np.cos(angle)
    s = np.sin(angle)
    t_x = x_ * c - y_ * s
    t_y = x_ * s + y_ * c
    x = t_x + origin_x
    y = t_y + origin_y
    return x, y


def scale_pt_new_joint(x, y, origin, dist, scaling_const=[0.8, 4]):
    v = (x - origin[0], y - origin[1])

    length = np.sqrt(v[0] ** 2 + v[1] ** 2)

    u = (v[0] / length, v[1] / length)

    # x_, y_ = x+dist*u[0]/scaling_const[0], y+dist*u[1]/scaling_const[1]

    x_, y_ = x + dist * u[0] / scaling_const[0], y

    return x_, y_


# def rotate_image(image, angle, pt):
#     # image_center = tuple(np.array(image.shape[1::-1]) / 2)
#     rot_mat = cv2.getRotationMatrix2D(pt, angle, 1.0)
#     result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#     return result


# def get_base_animal_parts_1(fname, annot_file="./parts_annot/cat_parts_annot.json"):
#     keypoints, keypoints_dict, vis = get_keypoints(fname)

#     with open(annot_file) as f:
#         annots = json.load(f)

#     cats = {}
#     for i in range(len(annots["categories"])):
#         cats[annots["categories"][i]["id"]] = annots["categories"][i]["name"].strip()

#     annots_limbs = {}

#     for i in range(len(annots["annotations"])):
#         if annots["annotations"][i]["image_id"] == 2:
#             annots_limbs[cats[annots["annotations"][i]["category_id"]]] = annots[
#                 "annotations"
#             ][i]["segmentation"][0]

#     x_dict = {}
#     y_dict = {}
#     mid_pt_dict = {}
#     angle_dict = {}

#     for i in range(5, 13):
#         try:
#             xs, ys = get_xs_ys(annots_limbs[cats[i]])
#         except:
#             continue
#         keypts = [keypoints_dict[i.split()[0]] for i in cats[i].split("-")]

#         mid_pt = ((keypts[0][0] + keypts[1][0]) / 2, (keypts[0][1] + keypts[1][1]) / 2)
#         angle = np.arctan2(keypts[0][1] - keypts[1][1], keypts[0][0] - keypts[1][0])

#         xs_rotated = []
#         ys_rotated = []

#         # for x, y in zip(xs, ys):
#         #     t_x, t_y = rotate_about_pt(x, y, mid_pt[0], mid_pt[1], -angle)
#         #     xs_rotated.append(t_x)
#         #     ys_rotated.append(t_y)

#         # x_dict[i] = xs_rotated
#         # y_dict[i] = ys_rotated

#         x_dict[i] = xs
#         y_dict[i] = ys
#         mid_pt_dict[i] = mid_pt
#         angle_dict[i] = angle

#     image = Image.new("RGB", (512, 512))

#     draw = ImageDraw.Draw(image)
#     points = [(xi, yi) for (xi, yi) in zip(x_dict[8], y_dict[8])]
#     draw.polygon((points), fill="#ffffff")

#     image.save("mask.png")

#     img = cv2.imread(
#         "/media2/het/Incremental_pose/data/cropped_images/2009_000553_1.jpg"
#     )

#     mask = cv2.imread("mask.png")

#     fg = cv2.bitwise_and(img, mask)

#     cv2.imwrite("masked_part.png", fg)
#     cv2.imwrite("temp1.png", img)

#     img = cv2.imread("masked_part.png")

#     rot_img = rotate_image(img, -(angle_dict[8] * 57.2958), mid_pt_dict[8])

#     cv2.imwrite("rotated_masked_part.png", rot_img)

#     return x_dict, y_dict, mid_pt_dict, cats


# def get_base_animal_parts(fname, annot_file="./parts_annot/cat_parts_annot.json"):
#     keypoints, keypoints_dict, vis = get_keypoints(fname)

#     with open(annot_file) as f:
#         annots = json.load(f)

#     cats = {}
#     for i in range(len(annots["categories"])):
#         cats[annots["categories"][i]["id"]] = annots["categories"][i]["name"].strip()

#     annots_limbs = {}

#     for i in range(len(annots["annotations"])):
#         annots_limbs[cats[annots["annotations"][i]["category_id"]]] = annots[
#             "annotations"
#         ][i]["segmentation"][0]

#     x_dict = {}
#     y_dict = {}
#     x_rotated_dict = {}
#     y_rotated_dict = {}
#     mid_pt_dict = {}

#     for i in range(1, 9):
#         try:
#             xs, ys = get_xs_ys(annots_limbs[cats[i]])
#         except:
#             continue
#         keypts = [keypoints_dict[i.split()[0]] for i in cats[i].split("-")]

#         mid_pt = ((keypts[0][0] + keypts[1][0]) / 2, (keypts[0][1] + keypts[1][1]) / 2)
#         angle = np.arctan2(keypts[0][1] - keypts[1][1], keypts[0][0] - keypts[1][0])

#         xs_rotated = []
#         ys_rotated = []

#         for x, y in zip(xs, ys):
#             t_x, t_y = rotate_about_pt(x, y, mid_pt[0], mid_pt[1], -angle)
#             xs_rotated.append(t_x)
#             ys_rotated.append(t_y)

#         x_rotated_dict[i] = xs_rotated
#         y_rotated_dict[i] = ys_rotated

#         x_dict[i] = xs
#         y_dict[i] = ys
#         mid_pt_dict[i] = mid_pt

#     for i in range(1, 9):
#         image = Image.new("RGB", (512, 512))

#         draw = ImageDraw.Draw(image)

#         points = [(xi, yi) for (xi, yi) in zip(x_dict[i], y_dict[i])]
#         draw.polygon((points), fill="#ffffff")

#         image.save("mask.png")

#         img = cv2.imread(
#             "/media2/het/Incremental_pose/data/cropped_images/2009_000553_1.jpg"
#         )

#         mask = cv2.imread("mask.png")

#         fg = cv2.bitwise_and(img, mask)

#         cv2.imwrite("./masks/masked_part_{}.png".format(i), fg)

#     return x_dict, y_dict, x_rotated_dict, y_rotated_dict, mid_pt_dict, cats


# def overlay_parts_on_new_image(
#     fname, x_dict, y_dict, x_rotated_dict, y_rotated_dict, mid_pt_dict, cats
# ):

#     dog_keypoints, dog_keypoints_dict, dog_vis = get_keypoints(fname)

#     keypoints_name_dict = {}

#     keypoints_name_dict["L_Eye"] = 0
#     keypoints_name_dict["R_Eye"] = 1
#     keypoints_name_dict["Nose"] = 2
#     keypoints_name_dict["L_Ear"] = 3
#     keypoints_name_dict["R_Ear"] = 4
#     keypoints_name_dict["LF_Elbow"] = 5
#     keypoints_name_dict["LF_Paw"] = 6
#     keypoints_name_dict["RF_Paw"] = 7
#     keypoints_name_dict["RF_Elbow"] = 8
#     keypoints_name_dict["LB_Paw"] = 9
#     keypoints_name_dict["RB_Paw"] = 10
#     keypoints_name_dict["LB_Elbow"] = 11
#     keypoints_name_dict["RB_Elbow"] = 12
#     keypoints_name_dict["LF_Knee"] = 13
#     keypoints_name_dict["RF_Knee"] = 14
#     keypoints_name_dict["LB_Knee"] = 15
#     keypoints_name_dict["RB_Knee"] = 16

#     dog_xs_dict = {}
#     dog_ys_dict = {}

#     for i in range(1, 9):
#         keypoint_names = [j.split()[0] for j in cats[i].split("-")]

#         if len(keypoint_names) == 1:
#             continue

#         if (
#             dog_vis[keypoints_name_dict[keypoint_names[0]]] == 0
#             or dog_vis[keypoints_name_dict[keypoint_names[1]]] == 0
#         ):
#             continue

#         dog_keypts = [dog_keypoints_dict[j] for j in keypoint_names]

#         dog_mid_pt = [
#             (dog_keypts[0][0] + dog_keypts[1][0]) / 2,
#             (dog_keypts[0][1] + dog_keypts[1][1]) / 2,
#         ]

#         dog_angle = np.arctan2(
#             dog_keypts[0][1] - dog_keypts[1][1], dog_keypts[0][0] - dog_keypts[1][0]
#         )

#         dog_xs, dog_ys = [], []

#         pt1_dog, pt2_dog = (
#             [dog_keypts[0][0], dog_keypts[0][1]],
#             [dog_keypts[1][0], dog_keypts[1][1]],
#         )

#         dist2 = np.sqrt(
#             (pt1_dog[0] - dog_mid_pt[0]) ** 2 + (pt1_dog[1] - dog_mid_pt[1]) ** 2
#         )  # Helps in finding the scaling factor of the polygon limb

#         new_mid_pt = (
#             dog_mid_pt[0] - mid_pt_dict[i][0],
#             dog_mid_pt[1] - mid_pt_dict[i][1],
#         )

#         for x, y in zip(x_rotated_dict[i], y_rotated_dict[i]):
#             t_x, t_y = (
#                 x + new_mid_pt[0],
#                 y + new_mid_pt[1],
#             )  # Shift the points to new mid point
#             t_x, t_y = scale_pt_new_joint(
#                 t_x, t_y, dog_mid_pt, dist2, [2, 1]
#             )  # Scale the polygon only on x-axis
#             t_x, t_y = rotate_about_pt(
#                 t_x, t_y, dog_mid_pt[0], dog_mid_pt[1], dog_angle,
#             )  # Rotate the polygon about the mid_point of the limb in the new image
#             if (
#                 t_x > 512
#             ):  # Simple conditions to make sure the coordinates are within the bounds
#                 t_x = 512
#             if t_y > 512:
#                 t_y = 512
#             if t_x < 0:
#                 t_x = 0
#             if t_y < 0:
#                 t_y = 0
#             dog_xs.append(t_x)
#             dog_ys.append(t_y)

#         dog_xs_dict[i] = dog_xs
#         dog_ys_dict[i] = dog_ys

#     dst = 0

#     parts = list(dog_xs_dict.keys())

#     print(parts)

#     if len(parts) == 0:
#         return

#     i = parts[0]

#     pts1 = [[x, y] for x, y in zip(x_dict[i], y_dict[i])]
#     pts2 = [[x, y] for x, y in zip(dog_xs_dict[i], dog_ys_dict[i])]

#     pts1 = np.float32(pts1[:-1])
#     pts2 = np.float32(pts2[:-1])

#     img = cv2.imread("./masks/masked_part_{}.png".format(i))

#     M = cv2.getPerspectiveTransform(pts1, pts2)

#     dst = cv2.bitwise_or(dst, cv2.warpPerspective(img, M, (512, 512)))

#     cv2.imwrite("transformed_part.png", dst)

#     for i in parts[1:]:
#         print(i)
#         pts1 = [[x, y] for x, y in zip(x_dict[i], y_dict[i])]
#         pts2 = [[x, y] for x, y in zip(dog_xs_dict[i], dog_ys_dict[i])]

#         pts1 = np.float32(pts1[:-1])
#         pts2 = np.float32(pts2[:-1])

#         img = cv2.imread("./masks/masked_part_{}.png".format(i))

#         M = cv2.getPerspectiveTransform(pts1, pts2)

#         dst = cv2.warpPerspective(img, M, (512, 512))

#         img_b = cv2.imread("transformed_part.png")
#         ret, thresh = cv2.threshold(img_b, 10, 255, cv2.THRESH_BINARY)

#         rows, cols, channels = img_b.shape
#         roi = dst[0:rows, 0:cols]

#         img2gray = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
#         ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
#         mask_inv = cv2.bitwise_not(mask)

#         img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
#         img2_fg = cv2.bitwise_or(img_b, img_b, mask=mask)

#         dst_new = cv2.add(img1_bg, img2_fg)
#         dst[0:rows, 0:cols] = dst_new
#         cv2.imwrite("transformed_part.png", dst)

#     dog_img = cv2.imread(
#         "/media2/het/Incremental_pose/data/cropped_images/" + fname[:-4] + ".jpg"
#     )

#     mask = np.full((512, 512), 0, dtype=np.uint8)

#     img2gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#     ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
#     mask_inv = cv2.bitwise_not(mask)

#     img1_bg = cv2.bitwise_and(dog_img, dog_img, mask=mask_inv)

#     img2_fg = cv2.bitwise_and(dst, dst, mask=mask)

#     final = cv2.add(img1_bg, img2_fg)

#     cv2.imwrite(
#         # "/media2/het/Incremental_pose/data/parts_overlayed/" + fname[:-4] + ".jpg",
#         "final.png",
#         final,
#     )


# (
#     x_dict,
#     y_dict,
#     x_rotated_dict,
#     y_rotated_dict,
#     mid_pt_dict,
#     cats,
# ) = get_base_animal_parts("2009_000553_1.xml", "temp_annot.json")

# overlay_parts_on_new_image(
#     "2010_003239_3_fn.xml",
#     x_dict,
#     y_dict,
#     x_rotated_dict,
#     y_rotated_dict,
#     mid_pt_dict,
#     cats,
# )


# def overlay_parts_on_new_image_1(
#     fname, x_dict_parts, y_dict_parts, mid_pt_dict_parts, cats
# ):
#     dog_keypoints, dog_keypoints_dict, dog_vis = get_keypoints(fname)

#     keypoints_name_dict = {}

#     keypoints_name_dict["L_Eye"] = 0
#     keypoints_name_dict["R_Eye"] = 1
#     keypoints_name_dict["Nose"] = 2
#     keypoints_name_dict["L_Ear"] = 3
#     keypoints_name_dict["R_Ear"] = 4
#     keypoints_name_dict["LF_Elbow"] = 5
#     keypoints_name_dict["LF_Paw"] = 6
#     keypoints_name_dict["RF_Paw"] = 7
#     keypoints_name_dict["RF_Elbow"] = 8
#     keypoints_name_dict["LB_Paw"] = 9
#     keypoints_name_dict["RB_Paw"] = 10
#     keypoints_name_dict["LB_Elbow"] = 11
#     keypoints_name_dict["RB_Elbow"] = 12
#     keypoints_name_dict["LF_Knee"] = 13
#     keypoints_name_dict["RF_Knee"] = 14
#     keypoints_name_dict["LB_Knee"] = 15
#     keypoints_name_dict["RB_Knee"] = 16

#     dog_xs_dict = {}
#     dog_ys_dict = {}

#     for i in range(5, 13):
#         keypoint_names = [j.split()[0] for j in cats[i].split("-")]

#         if len(keypoint_names) == 1:
#             continue

#         if (
#             dog_vis[keypoints_name_dict[keypoint_names[0]]] == 0
#             or dog_vis[keypoints_name_dict[keypoint_names[1]]] == 0
#         ):
#             continue

#         dog_keypts = [dog_keypoints_dict[j] for j in keypoint_names]

#         dog_mid_pt = [
#             (dog_keypts[0][0] + dog_keypts[1][0]) / 2,
#             (dog_keypts[0][1] + dog_keypts[1][1]) / 2,
#         ]

#         dog_angle = np.arctan2(
#             dog_keypts[0][1] - dog_keypts[1][1], dog_keypts[0][0] - dog_keypts[1][0]
#         )

#         dog_xs, dog_ys = [], []

#         pt1_dog, pt2_dog = (
#             [dog_keypts[0][0], dog_keypts[0][1]],
#             [dog_keypts[1][0], dog_keypts[1][1]],
#         )

#         dist2 = np.sqrt(
#             (pt1_dog[0] - dog_mid_pt[0]) ** 2 + (pt1_dog[1] - dog_mid_pt[1]) ** 2
#         )  # Helps in finding the scaling factor of the polygon limb

#         new_mid_pt = (
#             dog_mid_pt[0] - mid_pt_dict_parts[i][0],
#             dog_mid_pt[1] - mid_pt_dict_parts[i][1],
#         )

#         for x, y in zip(x_dict_parts[i], y_dict_parts[i]):
#             t_x, t_y = (
#                 x + new_mid_pt[0],
#                 y + new_mid_pt[1],
#             )  # Shift the points to new mid point
#             t_x, t_y = scale_pt_new_joint(
#                 t_x, t_y, dog_mid_pt, dist2, [2, 1]
#             )  # Scale the polygon only on x-axis
#             t_x, t_y = rotate_about_pt(
#                 t_x, t_y, dog_mid_pt[0], dog_mid_pt[1], dog_angle,
#             )  # Rotate the polygon about the mid_point of the limb in the new image
#             if (
#                 t_x > 512
#             ):  # Simple conditions to make sure the coordinates are within the bounds
#                 t_x = 512
#             if t_y > 512:
#                 t_y = 512
#             if t_x < 0:
#                 t_x = 0
#             if t_y < 0:
#                 t_y = 0
#             dog_xs.append(t_x)
#             dog_ys.append(t_y)

#         dog_xs_dict[i] = dog_xs
#         dog_ys_dict[i] = dog_ys

#     return dog_xs_dict, dog_ys_dict


# x_dict, y_dict, mid_pt_dict, cats = get_base_animal_parts("2009_000553_1.xml")
# dog_xs_dict, dog_ys_dict = overlay_parts_on_new_image(
#     "2010_004204_1.xml", x_dict, y_dict, mid_pt_dict, cats
# )

# image = Image.open("/media2/het/Incremental_pose/data/cropped_images/2010_004204_1.jpg")

# draw = ImageDraw.Draw(image)
# for key in dog_xs_dict.keys():

#     points = [(xi, yi) for (xi, yi) in zip(dog_xs_dict[key], dog_ys_dict[key])]
#     draw.polygon((points), fill="#ffffff")
# # draw.ellipse((pt1_dog[0]-3, pt1_dog[1]-3, pt1_dog[0]+3, pt1_dog[1]+3), fill=(255,0,0,0))
# # draw.ellipse((pt2_dog[0]-3, pt2_dog[1]-3, pt2_dog[0]+3, pt2_dog[1]+3), fill=(255,0,0,0))

# image.save("temp.png")


# def generate_part_augmentation(
#     cat_parts_fname="2009_000553_1.xml",
#     cat_parts_annot_name="./parts_annot/cat_parts_annot.json",
#     animal_class_to_be_augmented="dog",
# ):
#     annot_df = pd.read_csv("/media2/het/Incremental_pose/data/updated_df.csv")

#     animal_fnames = list(
#         annot_df.loc[annot_df["class"] == animal_class_to_be_augmented]["filename"]
#     )

#     x_dict, y_dict, mid_pt_dict, cats = get_base_animal_parts(
#         cat_parts_fname, cat_parts_annot_name
#     )

#     for f in animal_fnames:
#         dog_xs_dict, dog_ys_dict = overlay_parts_on_new_image(
#             f, x_dict, y_dict, mid_pt_dict, cats
#         )

#         image = Image.open(
#             "/media2/het/Incremental_pose/data/cropped_images/" + f[:-4] + ".jpg"
#         )

#         draw = ImageDraw.Draw(image)
#         for key in dog_xs_dict.keys():
#             points = [(xi, yi) for (xi, yi) in zip(dog_xs_dict[key], dog_ys_dict[key])]
#             draw.polygon((points), fill="#ffffff")

#         image.save(
#             "/media2/het/Incremental_pose/data/parts_overlayed/" + f[:-4] + ".jpg"
#         )


# generate_part_augmentation(animal_class_to_be_augmented="dog")
# generate_part_augmentation(animal_class_to_be_augmented="cow")
# generate_part_augmentation(animal_class_to_be_augmented="horse")

# annot_df = pd.read_csv("/media2/het/Incremental_pose/data/updated_df.csv")

# fname_list = list(annot_df.loc[annot_df["class"] == "dog"]["filename"])

# ones = np.ones((17,), dtype=np.int32)

# for f in fname_list:
#     _, _, vis = get_keypoints(f)
#     vis = np.array(vis)

#     if np.all(vis == ones):
#         print(f)

# img = Image.open("/media2/het/Incremental_pose/data/cropped_images/2008_001550_1.jpg")

# keypoints, _, _ = get_keypoints("2008_001550_1.xml")

# print(keypoints)

# draw = ImageDraw.Draw(img)

# for k in keypoints:
#     draw.ellipse((k[0] - 3, k[1] - 3, k[0] + 3, k[1] + 3), fill=(255, 0, 0, 0))

# img.save("temp.png")

# 2009_000553_1.xml => cat
# 2008_001550_1.xml => dog
# 2009_003825_1.xml => cow
# 2011_002730_1.xml => horse


generate_part_augmentation(
    "2011_002730_1.xml",
    "./parts_annot/horse_parts_annot.json",
    "horse",
    "/media2/het/Incremental_pose/data/parts_overlayed_same_class/",
)

generate_part_augmentation(
    "2008_001550_1.xml",
    "./parts_annot/dog_parts_annot.json",
    "dog",
    "/media2/het/Incremental_pose/data/parts_overlayed_same_class/",
)

generate_part_augmentation(
    "2009_003825_1.xml",
    "./parts_annot/cow_parts_annot.json",
    "cow",
    "/media2/het/Incremental_pose/data/parts_overlayed_same_class/",
)

generate_part_augmentation(
    "2009_000553_1.xml",
    "./parts_annot/cat_parts_annot.json",
    "cat",
    "/media2/het/Incremental_pose/data/parts_overlayed_same_class/",
)

