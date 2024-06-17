from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import Dataset


class CriteoDataset(Dataset):
    def __init__(self, file, read_part=True, sample_num=100000):
        super(CriteoDataset, self).__init__()

        names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
                 'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
                 'C23', 'C24', 'C25', 'C26']

        if read_part: # 默认读取 100000
            data_df = pd.read_csv(file, sep='\t', header=None, names=names, nrows=sample_num)
        else:
            data_df = pd.read_csv(file, sep='\t', header=None, names=names)

        sparse_features = ['C' + str(i) for i in range(1, 27)]
        dense_features = ['I' + str(i) for i in range(1, 14)]
        features = sparse_features + dense_features

        # 缺失值填充
        data_df[sparse_features] = data_df[sparse_features].fillna('-1')
        data_df[dense_features] = data_df[dense_features].fillna(0)

        # 连续型特征等间隔分箱
        est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
        data_df[dense_features] = est.fit_transform(data_df[dense_features])

        # 离散型特征转换成连续数字，为了在与参数计算时使用索引的方式计算，而不是向量乘积
        data_df[features] = OrdinalEncoder().fit_transform(data_df[features])

        self.data = data_df[features + ['label']].values

    # dataset 必须实现的方法
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][:-1], dtype=torch.long), torch.tensor([self.data[idx][-1]], dtype=torch.float)
         
    def get_field_dims(self):
        field_dims = (self.data.max(axis=0).astype(int) + 1).tolist()[:-1]
        return field_dims
        
    def train_valid_test_split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        field_dims = (self.data.max(axis=0).astype(int) + 1).tolist()[:-1]

        train, valid_test = train_test_split(self.data, train_size=train_size, random_state=2021)

        valid_size = valid_size / (test_size + valid_size)
        valid, test = train_test_split(valid_test, train_size=valid_size, random_state=2021)

        device = self.device

        train_X = torch.tensor(train[:, :-1], dtype=torch.long).to(device)
        valid_X = torch.tensor(valid[:, :-1], dtype=torch.long).to(device)
        test_X = torch.tensor(test[:, :-1], dtype=torch.long).to(device)
        train_y = torch.tensor(train[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        valid_y = torch.tensor(valid[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        test_y = torch.tensor(test[:, -1], dtype=torch.float).unsqueeze(1).to(device)

        return field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y)

