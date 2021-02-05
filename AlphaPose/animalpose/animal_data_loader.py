''' Dataloader for animal data ''' 
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torchvision
import torchvision.transforms
import torch.utils.data as data

class RandomFlip(object):
  def __call__(self, sample): 
    if np.random.uniform(0,1) > 0.5:
      sample['img'] = ImageOps.mirror(sample['img'])
      temp = []
      for k in sample['keypoints']:
        temp.append([[512-k[0][0], k[0][1]], [k[1][0], k[1][1]]])
      sample['keypoints'] = np.array(temp)
    return sample

class Noise(object):
  def __call__(self, sample):
    img = np.array(sample['img'])

    noise = np.random.randint(low=0,high=5, size = (512, 512, 3), dtype = 'uint8')

    for i in range(512):
        for j in range(512):
            for k in range(3):
                if (img[i][j][k] != 255):
                    img[i][j][k] += noise[i][j][k]

    sample['img'] =img

    return sample


class ToTensor(object):
  def __call__(self, sample):
    # print(sample['img'])
    img = np.array(sample['img']).transpose((2, 0, 1))
    sample['img'] = torch.FloatTensor(img)
    sample['keypoints'] = torch.FloatTensor(sample['keypoints'])
    # sample['img'] = torch.from_numpy(img, dtype=torch.float)
    # sample['keypoints'] = torch.from_numpy(sample['keypoints'], dtype=torch.float)
    return sample

class AnimalDatasetCombined(data.Dataset):
  def __init__(self, images_dir, annot_file, images_list, input_size, output_size, n_joints=17, train=True, transforms = None):
    self.keypoint_names = ['L_Eye', 'R_Eye', 'Nose', 'L_EarBase', 'R_EarBase',
                           'L_F_Elbow', 'L_F_Paw', 'R_F_Paw', 'R_F_Elbow',
                           'L_B_Paw', 'R_B_Paw', 'L_B_Elbow', 'R_B_Elbow',
                           'L_F_Knee', 'R_F_Knee', 'L_B_Knee', 'R_B_Knee']
    self.images_dir = images_dir
    self.annot_file = annot_file
    self._heatmap_size = output_size
    self.train = train
    self.transform = transforms
    self.n_joints = n_joints

    self.annot_df = pd.read_csv(self.annot_file)

    self.images = images_list
    # total_instances = len(self.images_list)
    # self.train_instances = int(0.9 * total_instances)

    # self.images_list = np.array(self.annot_df['filename'])
    # np.random.shuffle(self.images_list)

    # self.images = self.images_list
    # if self.train:
    #   self.images = self.images_list[:self.train_instances]
    # else:
    #   self.images = self.images_list[self.train_instances:]

    self._sigma = 2
    self._feat_stride = np.array(input_size) / np.array(output_size)

  def __len__(self):
    # if self.train:
    #   return self.train_instances
    # else:
    return len(self.images)

  def __getitem__(self, idx):
    img_name = os.path.join(self.images_dir, self.images[idx][:-4]+".jpg")
    img = Image.open(img_name)
    keypoints = []
    for keypt in self.keypoint_names:
      x = self.annot_df.loc[self.annot_df['filename'] == self.images[idx]][keypt+'_x'].item()
      y = self.annot_df.loc[self.annot_df['filename'] == self.images[idx]][keypt+'_y'].item()
      vis = self.annot_df.loc[self.annot_df['filename'] == self.images[idx]][keypt+'_visible'].item()
      keypoints.append([[x,vis],[y,vis]])
    keypoints = np.array(keypoints)

    if self.transform:
      sample = self.transform({'img': img, 'keypoints': keypoints})
      img = sample['img']
      keypoints = sample['keypoints']

    label, label_mask = self.label_transformer(img, keypoints)
    return img, label, label_mask

  def label_transformer(self, img, keypoints):
    target_weight = np.ones((self.n_joints, 1), dtype=np.float32)
    target_weight[:, 0] = keypoints[:, 0, 1]
    target = np.zeros((self.n_joints, self._heatmap_size[0], self._heatmap_size[1]),
                      dtype=np.float32)
    tmp_size = self._sigma * 3
    for i in range(self.n_joints):
      mu_x = int(keypoints[i, 0, 0] / self._feat_stride[0] + 0.5)
      mu_y = int(keypoints[i, 1, 0] / self._feat_stride[1] + 0.5)
      # check if any part of the gaussian is in-bounds
      ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
      br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
      if (ul[0] >= self._heatmap_size[1] or ul[1] >= self._heatmap_size[0] or br[0] < 0 or br[1] < 0):
        # return image as is
        target_weight[i] = 0
        continue
      # generate gaussian
      size = 2 * tmp_size + 1
      x = np.arange(0, size, 1, np.float32)
      y = x[:, np.newaxis]
      x0 = y0 = size // 2
      # the gaussian is not normalized, we want the center value to be equal to 1
      g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self._sigma ** 2)))
      # usable gaussian range
      g_x = max(0, -ul[0]), min(br[0], self._heatmap_size[1]) - ul[0]
      g_y = max(0, -ul[1]), min(br[1], self._heatmap_size[0]) - ul[1]
      # image range
      img_x = max(0, ul[0]), min(br[0], self._heatmap_size[1])
      img_y = max(0, ul[1]), min(br[1], self._heatmap_size[0])
      v = target_weight[i]
      if v > 0.5:
        target[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    target_weight = np.expand_dims(target_weight, -1)
    target = torch.FloatTensor(target)
    target_weight = torch.FloatTensor(target_weight)
    return target, target_weight
