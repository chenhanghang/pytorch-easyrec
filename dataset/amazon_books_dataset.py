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


class AmazonBooksDataset(Dataset):
    def __init__(self, file, read_part=True, sample_num=100000,sequence_length=40):
        super(AmazonBooksDataset, self).__init__()
        if read_part:
            data_df = pd.read_csv(file, sep=',', nrows=sample_num)
        else:
            data_df = pd.read_csv(file, sep=',')

        data_df['hist_item_list'] = data_df.apply(lambda x: x['hist_item_list'].split('|'), axis=1)
        data_df['hist_cate_list'] = data_df.apply(lambda x: x['hist_cate_list'].split('|'), axis=1)

        # cate encoder
        cate_list = list(data_df['cateID']) # cate 列 所有id
        data_df.apply(lambda x: cate_list.extend(x['hist_cate_list']), axis=1) # 历史中的cate
        cate_set = set(cate_list + ['0'])
        cate_encoder = LabelEncoder().fit(list(cate_set))
        self.cate_set = cate_encoder.transform(list(cate_set))

        # cate pad and transform
        hist_limit = sequence_length
        col = ['hist_cate_{}'.format(i) for i in range(hist_limit)]

        def deal(x):
            if len(x) > hist_limit:
                return pd.Series(x[-hist_limit:], index=col)
            else:
                pad = hist_limit - len(x)
                x = x + ['0' for _ in range(pad)]
                return pd.Series(x, index=col)

        cate_df = data_df['hist_cate_list'].apply(deal).join(data_df[['cateID']]).apply(cate_encoder.transform).join(
            data_df['label'])
        self.data = cate_df.values # 转 numpy

    # dataset 必须实现的方法
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][:-1], dtype=torch.long), torch.tensor([self.data[idx][-1]], dtype=torch.float)
         
    def get_field_dims(self):
        field_dims = [self.data[:-1].max().astype(int) + 1]
        return field_dims
        
    def train_valid_test_split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        field_dims = [self.data[:-1].max().astype(int) + 1]
        num_data = len(self.data)
        num_train = int(train_size * num_data)
        num_test = int(test_size * num_data)
        train = self.data[:num_train]
        valid = self.data[num_train: -num_test]
        test = self.data[-num_test:]

        device = self.device
        train_X = torch.tensor(train[:, :-1], dtype=torch.long).to(device)
        valid_X = torch.tensor(valid[:, :-1], dtype=torch.long).to(device)
        test_X = torch.tensor(test[:, :-1], dtype=torch.long).to(device)
        train_y = torch.tensor(train[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        valid_y = torch.tensor(valid[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        test_y = torch.tensor(test[:, -1], dtype=torch.float).unsqueeze(1).to(device)

        return field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y)

