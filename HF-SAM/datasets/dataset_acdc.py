import itertools
import os
import random
import re
from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from skimage import io
import cv2
class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', list_dir=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        train_ids, val_ids, test_ids = self._get_ids()
        if self.split.find('train') != -1:
            self.all_slices = os.listdir(
                self._base_dir + "/train_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split.find('val') != -1:
            self.all_volumes = os.listdir(
                self._base_dir + "/test_vol")
            self.sample_list = []
            for ids in val_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        elif self.split.find('test') != -1:
            self.all_volumes = os.listdir(
                self._base_dir )
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_ids(self):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        testing_set = ["patient{:0>3}".format(i) for i in range(1, 21)]
        validation_set = ["patient{:0>3}".format(i) for i in range(21, 31)]
        training_set = [i for i in all_cases_set if i not in testing_set+validation_set]

        return [training_set, validation_set, testing_set]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        # image = h5f['image'][:]
        # label = h5f['label'][:]
        # sample = {'image': image, 'label': label}
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/train_slices/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]  # fix sup_type to label
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self._base_dir + "/{}".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            sample = {'image': image, 'label': label}
        sample["idx"] = idx
        sample['case_name'] = case.replace('.h5', '')
        return sample


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


class RandomGenerator2(object):
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



class RandomGenerator(object):
    def __init__(self, output_size, low_res=None):
        """
        output_size: [H, W] 训练输入/输出分辨率
        low_res:     [h_low, w_low] 低分辨率标签大小（例如 [112,112]）。
                     若为 None，则不返回 low_res_label。
        """
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 原有随机增强（保持不变）
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # 尺度统一到 output_size（保持不变）
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # 图像用三次插值
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # 标签用最近邻

        # === 新增：生成低分辨率标签（仅在需要时） ===
        low_res_label = None
        if self.low_res is not None:
            H, W = self.output_size
            h_low, w_low = self.low_res
            # 对 label 做最近邻下采样，避免类别被插值破坏
            low_res_label = zoom(label, (h_low / H, w_low / W), order=0)

        # 打包张量（保持不变）
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        sample_out = {'image': image, 'label': label.long()}
        if low_res_label is not None:
            sample_out['low_res_label'] = torch.from_numpy(low_res_label.astype(np.int64))
        return sample_out
    


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
