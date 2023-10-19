import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

import dataset.provider as provider
from dataset.PCAlign import PCAlign
from dataset.datasset_utils import farthest_point_sample

warnings.filterwarnings('ignore')


def my_collate_fn(batch, train):
    batch_point_data = []
    batch_point_label = []
    for point_data, point_label in batch:
        batch_point_data.append(point_data)
        batch_point_label.append(point_label)
    batch_point_data = np.array(batch_point_data)
    batch_point_label = np.array(batch_point_label)
    if train:
        # --------------------------------------------------------------------------------------------------------------
        # 数据增强
        # --------------------------------------------------------------------------------------------------------------
        batch_point_data = provider.random_point_dropout(batch_point_data)
        batch_point_data[:, :, 0:3] = provider.random_scale_point_cloud(batch_point_data[:, :, 0:3])
        batch_point_data[:, :, 0:3] = provider.shift_point_cloud(batch_point_data[:, :, 0:3])
    # ------------------------------------------------------------------------------------------------------------------
    # 数据归一化
    # ------------------------------------------------------------------------------------------------------------------
    batch_point_data = PCAlign(batch_point_data)
    batch_point_data[:, :, :, 0:3] = provider.normalize_data(batch_point_data[:, :, :, 0:3])
    # ------------------------------------------------------------------------------------------------------------------
    # 转tensor
    # ------------------------------------------------------------------------------------------------------------------
    batch_point_data = torch.from_numpy(batch_point_data).reshape(-1, batch_point_data.shape[2],
                                                                  batch_point_data.shape[3])
    batch_point_label = torch.from_numpy(batch_point_label)
    return batch_point_data, batch_point_label


class ModelNetDataset(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.root = args.data_path
        self.npoints = args.num_point
        self.uniform = args.use_uniform_sample
        self.num_category = args.num_category
        # --------------------------------------------------------------------------------------------------------------
        # 读取类别文件
        # --------------------------------------------------------------------------------------------------------------
        self.catfile = os.path.join(self.root, 'ModelNet{}_shape_names.txt'.format(args.num_category))
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # --------------------------------------------------------------------------------------------------------------
        # 读取点云文件
        # --------------------------------------------------------------------------------------------------------------
        shape_ids = {split: [line.rstrip() for line in
                             open(os.path.join(self.root, 'ModelNet{}_{}.txt').format(args.num_category, split))]}
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.data_path = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt')
                          for i in range(len(shape_ids[split]))]
        self.cache = {}
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_cloud_data, point_class_num = self.cache[index]
        else:
            class_name, file_path = self.data_path[index][0], self.data_path[index][1]
            point_cloud_data = np.loadtxt(file_path, delimiter=',').astype(np.float32)
            point_class_num = np.array([self.classes[class_name]]).astype(np.int32)
            # ----------------------------------------------------------------------------------------------------------
            # 最远点采样
            # ----------------------------------------------------------------------------------------------------------
            if self.uniform:
                choice = farthest_point_sample(point_cloud_data, self.npoints)
            else:
                choice = np.random.choice(len(point_cloud_data), self.npoints, replace=True)
            point_cloud_data = point_cloud_data[choice, :]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_cloud_data, point_class_num)
        return point_cloud_data, point_class_num

    def __len__(self):
        return len(self.data_path)
