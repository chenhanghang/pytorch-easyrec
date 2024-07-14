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
from models.base.layer import Dataset


class CcpDataset(Dataset):
    def __init__(self, model_name, data_path='./data/ali-ccp'):
        super(CcpDataset, self).__init__()

        self.df_train = pd.read_csv(data_path + '/ali_ccp_train_sample.csv')
        self.df_val = pd.read_csv(data_path + '/ali_ccp_val_sample.csv')
        self.df_test = pd.read_csv(data_path + '/ali_ccp_test_sample.csv')

        self.train_idx, self.val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
        self.data = pd.concat([df_train, df_val, df_test], axis=0) # 所有训练数据
        #task 1 (as cvr): main task, purchase prediction
        #task 2(as ctr): auxiliary task, click prediction
        self.data.rename(columns={'purchase': 'cvr_label', 'click': 'ctr_label'}, inplace=True)
        self.data["ctcvr_label"] = self.data['cvr_label'] * data['ctr_label']
    
    # dataset 必须实现的方法
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][:-1], dtype=torch.long), torch.tensor([self.data[idx][-1]], dtype=torch.float)
         
    def get_field_dims(self):
        field_dims = (self.data.max(axis=0).astype(int) + 1).tolist()[:-1]
        return field_dims
        
    def train_valid_test_split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        col_names = self.data.columns.values.tolist()
        dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
        sparse_cols = [col for col in col_names if col not in dense_cols and col not in ['cvr_label', 'ctr_label', 'ctcvr_label']]
        #print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))
        #define dense and sparse features
        if model_name == "ESMM":
            label_cols = ['cvr_label', 'ctr_label', "ctcvr_label"]  #the order of 3 labels must fixed as this
            #ESMM only for sparse features in origin paper
            item_cols = ['129', '205', '206', '207', '210', '216']  #assumption features split for user and item
            user_cols = [col for col in sparse_cols if col not in item_cols]
            user_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in user_cols]
            item_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in item_cols]
            x_train, y_train = {name: data[name].values[:train_idx] for name in sparse_cols}, data[label_cols].values[:train_idx]
            x_val, y_val = {name: data[name].values[train_idx:val_idx] for name in sparse_cols}, data[label_cols].values[train_idx:val_idx]
            x_test, y_test = {name: data[name].values[val_idx:] for name in sparse_cols}, data[label_cols].values[val_idx:]
            return user_features, item_features, x_train, y_train, x_val, y_val, x_test, y_test
        else:
            label_cols = ['cvr_label', 'ctr_label']  #the order of labels can be any
            used_cols = sparse_cols + dense_cols
            features = [SparseFeature(col, data[col].max()+1, embed_dim=4)for col in sparse_cols] \
                       + [DenseFeature(col) for col in dense_cols]
            x_train, y_train = {name: data[name].values[:train_idx] for name in used_cols}, data[label_cols].values[:train_idx]
            x_val, y_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}, data[label_cols].values[train_idx:val_idx]
            x_test, y_test = {name: data[name].values[val_idx:] for name in used_cols}, data[label_cols].values[val_idx:]
            return features, x_train, y_train, x_val, y_val, x_test, y_test

        #return field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y)

