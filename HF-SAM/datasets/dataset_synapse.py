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
    
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split in ["train", "val"] or self.sample_list[idx].strip('\n').split(",")[0].endswith(".npz"):
            slice_name = self.sample_list[idx].strip('\n').split(",")[0]
            if slice_name.endswith(".npz"):
                data_path = os.path.join(self.data_dir, slice_name)
            else:
                data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            try:
                image, label = data['image'], data['label']
            except:
                image, label = data['data'], data['seg']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)

            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
