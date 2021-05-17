import os
import cv2
import json
import shutil
import random
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt

# import skimage
# from skimage.transform import PiecewiseAffineTransform, warp
# from skimage.io import imsave, imread

import torch
import torchvision

from animal_data_loader import AnimalDatasetCombined, ToTensor

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

import thinplate as tps
from ray.util.multiprocessing import Pool


def L2_dist(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.sqrt(np.abs(np.sum((v2 - v1) ** 2)))


def calc_dist(avg_vector, animal_list):
    result = []
    for i in animal_list:
        fname = i[0]
        keypoints = i[1]
        dist = L2_dist(avg_vector, keypoints)
        temp = ((fname, keypoints), dist)
        result.append(temp)
    return result


def ClusterIndicesNumpy(clustNum, labels_array):  # numpy
    return np.where(labels_array == clustNum)[0]


def save_images(
    fname_list,
    images_path="/media2/het/Incremental_pose/data/cropped_images/",
    annot_path="/media2/het/Incremental_pose/data/updated_df.csv",
    save_dir="./keypoints/",
    animal_class=None,
    train=True,
):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    trainset = AnimalDatasetCombined(
        images_path,
        annot_path,
        fname_list,
        input_size=(512, 512),
        output_size=(128, 128),
        transforms=torchvision.transforms.Compose([ToTensor()]),
        train=train,
    )

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

    for i, (inps, labels, label_masks, _) in enumerate(train_loader):
        for j in range(labels.shape[0]):
            final_img = np.zeros((128, 128))
            input_img = inps[j].cpu().numpy().transpose((1, 2, 0))

            input_img = Image.fromarray(input_img.astype(np.uint8))
            input_img.save(
                os.path.join(
                    save_dir, "image" + str(animal_class) + "_{}_{}.png".format(i, j)
                )
            )

            for k in range(labels.shape[1]):
                images = labels.cpu().numpy()[j][k]
                final_img += images
                # plt.imsave(
                #     os.path.join(
                #         save_dir,
                #         "temp" + str(animal_class) + "_{}_{}_{}.png".format(i, j, k),
                #     ),
                #     images,
                #     cmap="gray",
                # )
            plt.imsave(
                os.path.join(
                    save_dir, "temp" + str(animal_class) + "_{}_{}.png".format(i, j)
                ),
                final_img,
                cmap="gray",
            )


def get_rot_gaussian_maps(mu, shape_hw, inv_std1, inv_std2, angles, mode="rot"):
    """
	Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
	given the gaussian centers: MU [B, NMAPS, 2] tensor.

	STD: is the fixed standard dev.
	"""
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]  # (B, 12, 1)

    y = np.linspace(-1.0, 1.0, shape_hw[0])

    x = np.linspace(-1.0, 1.0, shape_hw[1])  # Bx14

    y = np.reshape(np.tile(y, [shape_hw[0]]), (-1, shape_hw[0], shape_hw[0]))
    y = np.expand_dims(y, 0) * np.ones((mu.shape[1], shape_hw[0], shape_hw[0]))

    x = np.reshape(
        np.tile(x, [shape_hw[1]]), (-1, shape_hw[1], shape_hw[1])
    )  # Bx128x128
    x = np.expand_dims(x, 0) * np.ones(
        (mu.shape[1], shape_hw[1], shape_hw[1])
    )  # Bx12x128x128
    x = np.transpose(x, [0, 1, 3, 2])
    mu_y, mu_x = np.expand_dims(mu_y, 3), np.expand_dims(mu_x, 3)  # Bx12x1x1

    y = y - mu_y
    x = x - mu_x  # Bx12x128x128

    if mode in ["rot", "flat"]:
        # apply rotation to the grid
        yx_stacked = np.stack(
            [
                np.reshape(y, (-1, y.shape[1], y.shape[2] * y.shape[3])),
                np.reshape(x, (-1, x.shape[1], x.shape[2] * x.shape[3])),
            ],
            2,
        )  # (B, 12, 2, 128^2)
        rot_mat = np.stack(
            [
                np.stack([np.cos(angles), np.sin(angles)], 2),
                np.stack([-np.sin(angles), np.cos(angles)], 2),
            ],
            3,
        )  # (B, 128, 2, 2)

        rotated = np.matmul(rot_mat, yx_stacked)  # (B, 12, 2, 128^2)

        y_rot = rotated[:, :, 0, :]  # (B, 12, 128^2)
        x_rot = rotated[:, :, 1, :]  # (B, 12, 128^2)

        y_rot = np.reshape(
            y_rot, (-1, mu.shape[1], shape_hw[0], shape_hw[0])
        )  # (B, 12, 128, 128)
        x_rot = np.reshape(
            x_rot, (-1, mu.shape[1], shape_hw[1], shape_hw[1])
        )  # (B, 12, 128, 128)

        g_y = np.square(y_rot)  # (B, 12, 128, 128)
        g_x = np.square(x_rot)  # (B, 12, 128, 128)

        inv_std1 = np.expand_dims(np.expand_dims(inv_std1, 2), 2)  # Bx12x1x1
        inv_std2 = np.expand_dims(np.expand_dims(inv_std2, 2), 2)  # Bx12x1x1
        dist = g_y * inv_std1 ** 2 + g_x * inv_std2 ** 2

        if mode == "rot":
            g_yx = np.exp(-dist)

        else:
            g_yx = np.exp(-np.pow(dist + 1e-5, 0.25))

    else:
        raise ValueError("Unknown mode: " + str(mode))

    g_yx = np.transpose(g_yx, [0, 1, 3, 2])
    g_yx = torch.tensor(g_yx, dtype=torch.float, requires_grad=False)

    return g_yx


def get_limb_centers(joints_2d, vis):
    limb_parents_dict = {
        0: [3],
        1: [4],
        2: [0, 1],
        13: [5],
        6: [13],
        14: [8],
        7: [14],
        15: [11],
        9: [15],
        16: [12],
        10: [16],
    }
    angles_x = []
    angles_y = []
    limbs_x = []
    limbs_y = []
    limb_length = []
    labels = []

    cnt = 0
    for i in limb_parents_dict.keys():
        for j in limb_parents_dict[i]:
            vis_pair = np.squeeze(
                np.stack(
                    [
                        np.logical_and(vis[:, i], vis[:, j]).astype(int),
                        np.logical_and(vis[:, i], vis[:, j]).astype(int),
                    ]
                )
            )

            temp_label = []

            for v in np.logical_and(vis[:, i], vis[:, j]):
                for t in v:
                    if t == 0:
                        temp_label.append(12)
                    else:
                        temp_label.append(cnt)

            labels.append(temp_label)
            cnt += 1
            x_pair = np.array([joints_2d[:, i, 0], joints_2d[:, j, 0]])
            x_pair = np.multiply(
                x_pair, vis_pair
            )  # Do not take limbs where keypoints are not visible

            y_pair = [joints_2d[:, i, 1], joints_2d[:, j, 1]]
            y_pair = np.multiply(y_pair, vis_pair)

            limbs_x.append((x_pair[0] + x_pair[1]) / 2.0)
            limbs_y.append((y_pair[0] + y_pair[1]) / 2.0)
            limb_length.append(
                np.sqrt((x_pair[0] - x_pair[1]) ** 2 + (y_pair[0] - y_pair[1]) ** 2)
            )

            angles_x.append(x_pair[1] - x_pair[0])  # because y is represented as x
            angles_y.append(y_pair[0] - y_pair[1])

    angles_x = np.stack(angles_x, 1)
    angles_y = np.stack(angles_y, 1)

    angles = np.arctan2(angles_x, angles_y + 1e-7)  # x/y as pose is passed as (y,x)

    limbs_x = np.stack(limbs_x, 1)
    limbs_y = np.stack(limbs_y, 1)

    limbs = np.stack([limbs_x, limbs_y], 2)
    limb_length = np.stack(limb_length, 1)

    labels = np.array(labels).T
    return limbs, angles, limb_length, labels


def limb_maps(pose_points, vis):
    points_exchanged = (
        np.stack([pose_points[:, :, 1], pose_points[:, :, 0]], 2) / 64.0 - 1.0
    )

    limb_centers_yx, angles, limb_length, labels = get_limb_centers(
        points_exchanged, vis
    )
    # decreasing the value of ratio increases the length of the gaussian
    length_ratios = []
    for i in limb_length:
        temp = []
        for j in i:
            if j:
                temp.append(5 / j)
            else:
                temp.append(0)
        length_ratios.append(temp)

    length_ratios = np.array(length_ratios)
    # decreasing the value of ratio increases the width of the gaussian
    width_ratios = np.array(
        [3.0, 3.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    ) * (np.ones_like(limb_length))

    gauss_map = get_rot_gaussian_maps(
        limb_centers_yx, [128, 128], width_ratios, length_ratios, angles, mode="rot"
    )

    labels = torch.FloatTensor(labels)

    return gauss_map, labels


def get_keypoints(fname, csv_file="/media2/het/Incremental_pose/data/updated_df.csv"):
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

    annot_df = pd.read_csv(csv_file)

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


def scale_pt_new_joint(x, y, origin, dist, scaling_const=[2, 1]):
    v = (x - origin[0], y - origin[1])

    length = np.sqrt(v[0] ** 2 + v[1] ** 2)

    u = (v[0] / length, v[1] / length)

    # x_, y_ = x+dist*u[0]/scaling_const[0], y+dist*u[1]/scaling_const[1]

    x_, y_ = x + dist * u[0] / scaling_const[0], y

    return x_, y_


def get_base_animal_parts(fname, annot_file="./parts_annot/cat_parts_annot.json"):
    keypoints, keypoints_dict, vis = get_keypoints(fname)

    with open(annot_file) as f:
        annots = json.load(f)

    cats = {}
    for i in range(len(annots["categories"])):
        cats[annots["categories"][i]["id"]] = annots["categories"][i]["name"].strip()

    annots_limbs = {}

    for i in range(len(annots["annotations"])):
        annots_limbs[cats[annots["annotations"][i]["category_id"]]] = annots[
            "annotations"
        ][i]["segmentation"][0]

    x_dict = {}
    y_dict = {}
    x_rotated_dict = {}
    y_rotated_dict = {}
    mid_pt_dict = {}

    for i in range(1, 9):
        try:
            xs, ys = get_xs_ys(annots_limbs[cats[i]])
        except:
            continue
        keypts = [keypoints_dict[i.split()[0]] for i in cats[i].split("-")]

        mid_pt = ((keypts[0][0] + keypts[1][0]) / 2, (keypts[0][1] + keypts[1][1]) / 2)
        angle = np.arctan2(keypts[0][1] - keypts[1][1], keypts[0][0] - keypts[1][0])

        xs_rotated = []
        ys_rotated = []

        for x, y in zip(xs, ys):
            t_x, t_y = rotate_about_pt(x, y, mid_pt[0], mid_pt[1], -angle)
            xs_rotated.append(t_x)
            ys_rotated.append(t_y)

        x_rotated_dict[i] = xs_rotated
        y_rotated_dict[i] = ys_rotated

        x_dict[i] = xs
        y_dict[i] = ys
        mid_pt_dict[i] = mid_pt

    for i in range(1, 9):
        image = Image.new("RGB", (512, 512))

        draw = ImageDraw.Draw(image)

        points = [(xi, yi) for (xi, yi) in zip(x_dict[i], y_dict[i])]
        draw.polygon((points), fill="#ffffff")

        image.save("mask.png")

        img = cv2.imread(
            "/media2/het/Incremental_pose/data/cropped_images/" + fname[:-4] + ".jpg"
        )

        mask = cv2.imread("mask.png")

        fg = cv2.bitwise_and(img, mask)

        cv2.imwrite("./masks/masked_part_{}.png".format(i), fg)

    return x_dict, y_dict, x_rotated_dict, y_rotated_dict, mid_pt_dict, cats


def overlay_parts_on_new_image(
    fname,
    x_dict,
    y_dict,
    x_rotated_dict,
    y_rotated_dict,
    mid_pt_dict,
    cats,
    save_dir="/media2/het/Incremental_pose/data/parts_overlayed/",
):

    dog_keypoints, dog_keypoints_dict, dog_vis = get_keypoints(fname)

    keypoints_name_dict = {}

    keypoints_name_dict["L_Eye"] = 0
    keypoints_name_dict["R_Eye"] = 1
    keypoints_name_dict["Nose"] = 2
    keypoints_name_dict["L_Ear"] = 3
    keypoints_name_dict["R_Ear"] = 4
    keypoints_name_dict["LF_Elbow"] = 5
    keypoints_name_dict["LF_Paw"] = 6
    keypoints_name_dict["RF_Paw"] = 7
    keypoints_name_dict["RF_Elbow"] = 8
    keypoints_name_dict["LB_Paw"] = 9
    keypoints_name_dict["RB_Paw"] = 10
    keypoints_name_dict["LB_Elbow"] = 11
    keypoints_name_dict["RB_Elbow"] = 12
    keypoints_name_dict["LF_Knee"] = 13
    keypoints_name_dict["RF_Knee"] = 14
    keypoints_name_dict["LB_Knee"] = 15
    keypoints_name_dict["RB_Knee"] = 16

    dog_xs_dict = {}
    dog_ys_dict = {}

    for i in range(1, 9):
        keypoint_names = [j.split()[0] for j in cats[i].split("-")]

        if len(keypoint_names) == 1:
            continue

        if (
            dog_vis[keypoints_name_dict[keypoint_names[0]]] == 0
            or dog_vis[keypoints_name_dict[keypoint_names[1]]] == 0
        ):
            continue

        dog_keypts = [dog_keypoints_dict[j] for j in keypoint_names]

        dog_mid_pt = [
            (dog_keypts[0][0] + dog_keypts[1][0]) / 2,
            (dog_keypts[0][1] + dog_keypts[1][1]) / 2,
        ]

        dog_angle = np.arctan2(
            dog_keypts[0][1] - dog_keypts[1][1], dog_keypts[0][0] - dog_keypts[1][0]
        )

        dog_xs, dog_ys = [], []

        pt1_dog, pt2_dog = (
            [dog_keypts[0][0], dog_keypts[0][1]],
            [dog_keypts[1][0], dog_keypts[1][1]],
        )

        dist2 = np.sqrt(
            (pt1_dog[0] - dog_mid_pt[0]) ** 2 + (pt1_dog[1] - dog_mid_pt[1]) ** 2
        )  # Helps in finding the scaling factor of the polygon limb

        new_mid_pt = (
            dog_mid_pt[0] - mid_pt_dict[i][0],
            dog_mid_pt[1] - mid_pt_dict[i][1],
        )

        for x, y in zip(x_rotated_dict[i], y_rotated_dict[i]):
            t_x, t_y = (
                x + new_mid_pt[0],
                y + new_mid_pt[1],
            )  # Shift the points to new mid point
            t_x, t_y = scale_pt_new_joint(
                t_x, t_y, dog_mid_pt, dist2, [2, 1]
            )  # Scale the polygon only on x-axis
            t_x, t_y = rotate_about_pt(
                t_x, t_y, dog_mid_pt[0], dog_mid_pt[1], dog_angle,
            )  # Rotate the polygon about the mid_point of the limb in the new image
            if (
                t_x > 512
            ):  # Simple conditions to make sure the coordinates are within the bounds
                t_x = 512
            if t_y > 512:
                t_y = 512
            if t_x < 0:
                t_x = 0
            if t_y < 0:
                t_y = 0
            dog_xs.append(t_x)
            dog_ys.append(t_y)

        dog_xs_dict[i] = dog_xs
        dog_ys_dict[i] = dog_ys

    dst = 0

    parts = list(dog_xs_dict.keys())

    if len(parts) == 0:
        return
    i = parts[0]

    pts1 = [[x, y] for x, y in zip(x_dict[i], y_dict[i])]
    pts2 = [[x, y] for x, y in zip(dog_xs_dict[i], dog_ys_dict[i])]

    pts1 = np.float32(pts1[:-1])
    pts2 = np.float32(pts2[:-1])

    img = cv2.imread("./masks/masked_part_{}.png".format(i))

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.bitwise_or(dst, cv2.warpPerspective(img, M, (512, 512)))

    cv2.imwrite("transformed_part.png", dst)

    if len(parts) > 1:
        for i in parts[1:]:
            pts1 = np.float32([[x, y] for x, y in zip(x_dict[i], y_dict[i])][:-1])
            pts2 = np.float32(
                [[x, y] for x, y in zip(dog_xs_dict[i], dog_ys_dict[i])][:-1]
            )

            img = cv2.imread("./masks/masked_part_{}.png".format(i))

            M = cv2.getPerspectiveTransform(pts1, pts2)

            dst = cv2.warpPerspective(img, M, (512, 512))

            img_b = cv2.imread("transformed_part.png")
            ret, thresh = cv2.threshold(img_b, 10, 255, cv2.THRESH_BINARY)

            rows, cols, channels = img_b.shape
            roi = dst[0:rows, 0:cols]

            img2gray = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            img2_fg = cv2.bitwise_or(img_b, img_b, mask=mask)

            dst_new = cv2.add(img1_bg, img2_fg)
            dst[0:rows, 0:cols] = dst_new
            cv2.imwrite("transformed_part.png", dst)

    dog_img = cv2.imread(
        "/media2/het/Incremental_pose/data/cropped_images/" + fname[:-4] + ".jpg"
    )

    dst = cv2.imread("transformed_part.png")

    mask = np.full((512, 512), 0, dtype=np.uint8)

    img2gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(dog_img, dog_img, mask=mask_inv)

    img2_fg = cv2.bitwise_and(dst, dst, mask=mask)

    final = cv2.add(img1_bg, img2_fg)

    cv2.imwrite(
        save_dir + fname[:-4] + ".jpg", final,
    )


def generate_part_augmentation(
    parts_fname="2009_000553_1.xml",  # 2009_000553_1.xml => cat, 2008_001550_1.xml => dog, 2009_003825_1.xml => cow, 2011_002730_1.xml => horse
    parts_annot_name="./parts_annot/cat_parts_annot.json",
    animal_class_to_be_augmented="dog",
    save_dir="/media2/het/Incremental_pose/data/parts_overlayed/",
):
    annot_df = pd.read_csv("/media2/het/Incremental_pose/data/updated_df.csv")

    animal_fnames = list(
        annot_df.loc[annot_df["class"] == animal_class_to_be_augmented]["filename"]
    )

    (
        x_dict,
        y_dict,
        x_rotated_dict,
        y_rotated_dict,
        mid_pt_dict,
        cats,
    ) = get_base_animal_parts(parts_fname, parts_annot_name)

    for f in tqdm(animal_fnames):
        overlay_parts_on_new_image(
            f,
            x_dict,
            y_dict,
            x_rotated_dict,
            y_rotated_dict,
            mid_pt_dict,
            cats,
            save_dir,
        )


def f(index):
    return index


def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)


def rotate_elbow_knee_limbs(
    input_dir="/media/gaurav/Incremental_pose/data/cropped_images/",
    output_dir="/media/gaurav/Incremental_pose/data/rotated_images/",
    input_csv_file="/media/gaurav/Incremental_pose/data/updated_df.csv",
    output_csv_file="/media/gaurav/Incremental_pose/data/updated_df_rotated.csv",
    animal_class=None,
):
    if not os.path.exists(output_dir):
        #     shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    # indices_to_be_rotated = [(5,13), (8,14), (11, 15), (12, 16)]
    # indices_not_to_be_rotated = [0, 1, 2, 3, 4, 6, 7, 9, 10]
    indices_to_be_rotated = [
        (5, 13, 6),
        # (5, 6),
        (8, 14, 7),
        # (8, 7),
        (11, 15, 9),
        # (11, 9),
        (12, 16, 10),
        # (12, 10),
    ]
    indices_not_to_be_rotated = [0, 1, 2, 3, 4]

    keypoints_dict = {}
    for i in range(17):
        keypoints_dict[i] = []

    df = pd.read_csv(input_csv_file)

    if animal_class:
        df = df.loc[df["class"] == animal_class]

    classes = list(df["class"])
    fname_list = list(df["filename"])

    cnt = 0
    # pool = Pool()

    # for fname in pool.map(f, fname_list):
    for fname in fname_list:
        print(fname)
        keypoints, _, vis = get_keypoints(fname, csv_file=input_csv_file)

        c_src = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]

        c_dst = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]

        # c_src = []
        # c_dst = []

        indices_visited = []

        for ind in indices_to_be_rotated:
            if vis[ind[0]] == 1 and vis[ind[1]] == 1 and vis[ind[2]] == 1:
                angle = random.uniform(0.25, 0.6)
                rot_pt_x, rot_pt_y = rotate_about_pt(
                    keypoints[ind[1]][0],
                    keypoints[ind[1]][1],
                    keypoints[ind[0]][0],
                    keypoints[ind[0]][1],
                    angle,
                )

                rot_pt_x_paw, rot_pt_y_paw = rotate_about_pt(
                    keypoints[ind[2]][0],
                    keypoints[ind[2]][1],
                    keypoints[ind[0]][0],
                    keypoints[ind[0]][1],
                    angle,
                )

                if rot_pt_x < 0 or rot_pt_y < 0 or rot_pt_x_paw < 0 or rot_pt_y_paw < 0:
                    rot_pt_x, rot_pt_y = keypoints[ind[1]][0], keypoints[ind[1]][1]
                    rot_pt_x_paw, rot_pt_y_paw = (
                        keypoints[ind[2]][0],
                        keypoints[ind[2]][1],
                    )

                    # move on
                    keypoints_dict[ind[0]].append(
                        [keypoints[ind[0]][0], keypoints[ind[0]][1], 1]
                    )
                    indices_visited.append(ind[0])

                    keypoints_dict[ind[1]].append([rot_pt_x, rot_pt_y, 1])
                    indices_visited.append(ind[1])

                    keypoints_dict[ind[2]].append([rot_pt_x_paw, rot_pt_y_paw, 1])
                    indices_visited.append(ind[2])
                    continue

                ## Original Keypoints
                c_src.append(
                    [keypoints[ind[0]][0] / 512.0, keypoints[ind[0]][1] / 512.0]
                )

                c_src.append(
                    [keypoints[ind[1]][0] / 512.0, keypoints[ind[1]][1] / 512.0]
                )

                c_src.append(
                    [keypoints[ind[2]][0] / 512.0, keypoints[ind[2]][1] / 512.0]
                )

                ## Keypoints after rotating them by a random angle
                c_dst.append(
                    [keypoints[ind[0]][0] / 512.0, keypoints[ind[0]][1] / 512.0]
                )
                c_dst.append([rot_pt_x / 512.0, rot_pt_y / 512.0])
                c_dst.append([rot_pt_x_paw / 512.0, rot_pt_y_paw / 512.0])

                if not ind[0] in indices_visited:
                    keypoints_dict[ind[0]].append(
                        [keypoints[ind[0]][0], keypoints[ind[0]][1], 1]
                    )
                    indices_visited.append(ind[0])

                if not ind[1] in indices_visited:
                    keypoints_dict[ind[1]].append([rot_pt_x, rot_pt_y, 1])
                    indices_visited.append(ind[1])

                if not ind[2] in indices_visited:
                    keypoints_dict[ind[2]].append([rot_pt_x_paw, rot_pt_y_paw, 1])
                    indices_visited.append(ind[2])

            else:
                if not ind[0] in indices_visited:
                    # keypoints_dict[ind[0]].append([0, 0, 0])
                    keypoints_dict[ind[0]].append(
                        [keypoints[ind[0]][0], keypoints[ind[0]][1], vis[ind[0]]]
                    )
                    indices_visited.append(ind[0])

                if not ind[1] in indices_visited:
                    # keypoints_dict[ind[0]].append([0, 0, 0])
                    keypoints_dict[ind[1]].append(
                        [keypoints[ind[1]][0], keypoints[ind[1]][1], vis[ind[1]]]
                    )
                    indices_visited.append(ind[1])

                if not ind[2] in indices_visited:
                    # keypoints_dict[ind[0]].append([0, 0, 0])
                    keypoints_dict[ind[2]].append(
                        [keypoints[ind[2]][0], keypoints[ind[2]][1], vis[ind[2]]]
                    )
                    indices_visited.append(ind[2])

        for ind in indices_not_to_be_rotated:
            if vis[ind] == 1:
                c_src.append([keypoints[ind][0] / 512.0, keypoints[ind][1] / 512.0])
                c_dst.append([keypoints[ind][0] / 512.0, keypoints[ind][1] / 512.0])
                keypoints_dict[ind].append([keypoints[ind][0], keypoints[ind][1], 1])

            else:
                keypoints_dict[ind].append([0, 0, 0])

        c_src = np.array(c_src)
        c_dst = np.array(c_dst)

        cols, rows = 512, 512

        img = cv2.imread(os.path.join(input_dir, fname[:-4] + ".jpg"))

        warped = warp_image_cv(img, c_src, c_dst, dshape=(512, 512))

        img2gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        dst = cv2.inpaint(warped, mask_inv, 3, cv2.INPAINT_TELEA)

        cv2.imwrite(os.path.join(output_dir, fname[:-4] + ".jpg"), dst)

        # except:
        #     img = cv2.imread(os.path.join(input_dir, fname[:-4] + ".jpg"))
        #     cv2.imwrite(os.path.join(output_dir, fname[:-4] + ".jpg"), img)

    indices_to_keypoints_dict = {
        0: "L_Eye",
        1: "R_Eye",
        2: "Nose",
        3: "L_EarBase",
        4: "R_EarBase",
        5: "L_F_Elbow",
        6: "L_F_Paw",
        7: "R_F_Paw",
        8: "R_F_Elbow",
        9: "L_B_Paw",
        10: "R_B_Paw",
        11: "L_B_Elbow",
        12: "R_B_Elbow",
        13: "L_F_Knee",
        14: "R_F_Knee",
        15: "L_B_Knee",
        16: "R_B_Knee",
    }

    data_dict = {}

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

    data_dict["filename"] = fname_list
    data_dict["class"] = classes

    for keypt in keypoint_names:
        data_dict[keypt + "_x"] = []
        data_dict[keypt + "_y"] = []
        data_dict[keypt + "_visible"] = []

    for i in keypoints_dict.keys():
        keypt = indices_to_keypoints_dict[i]
        for d in keypoints_dict[i]:
            data_dict[keypt + "_x"].append(d[0])
            data_dict[keypt + "_y"].append(d[1])
            data_dict[keypt + "_visible"].append(d[2])

    for k in data_dict.keys():
        print(f"Length of {k} keypoints list is : {len(data_dict[k])}")

    final_df = pd.DataFrame(data_dict)
    final_df.to_csv(output_csv_file, index=None)


class EarlyStopping(object):
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)
