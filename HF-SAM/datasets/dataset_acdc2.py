import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class ACDC_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        """
        兼容三种清单与存储：
        1) 清单中给出 .h5 文件名（优先）
        2) 清单中仅给出样本名 -> 默认拼接成 {data_dir}/{name}.h5
        3) 兼容历史 .npz（保留旧路径与字段名）
        """
        item = self.sample_list[idx].strip('\n').split(",")[0]

        # —— 优先：HDF5 读取 —— #
        # 如果清单里已经写了 .h5，直接按 .h5 读；否则默认拼上 .h5
        h5_path = item if item.endswith(".h5") else os.path.join(self.data_dir, item + ".h5")
        if os.path.exists(h5_path):
            # 建议使用 with 语法，确保及时关闭文件句柄
            with h5py.File(h5_path, "r") as f:
                # 约定 H5 内部键名为 'image' 与 'label'
                # 如你的键名不同（例如 'data'/'seg'），这里改成对应的键名即可
                image = f["image"][:]
                label = f["label"][:]
        else:
            # —— 兼容：NPZ 读取（保留旧逻辑） —— #
            # 支持清单里直接给 .npz 或只有样本名两种写法
            if item.endswith(".npz"):
                npz_path = os.path.join(self.data_dir, item)
            else:
                npz_path = os.path.join(self.data_dir, item + ".npz")
            data = np.load(npz_path)
            # 兼容两套键名
            try:
                image, label = data['image'], data['label']
            except KeyError:
                image, label = data['data'], data['seg']

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
