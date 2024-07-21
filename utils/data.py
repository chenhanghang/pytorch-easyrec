import random
import torch
import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import Dataset, DataLoader, random_split

class TorchDataset(Dataset):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        # map:{k,v},label
        return {k: v[index] for k, v in self.x.items()}, self.y[index]

    def __len__(self):
        return len(self.y)


class PredictDataset(Dataset):

    def __init__(self, x):
        super().__init__()
        self.x = x # map:{name: numpy_array}

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}

    def __len__(self):
        return len(self.x[list(self.x.keys())[0]])

class MatchDataGenerator(object):

    def __init__(self, x, y=[]):
        super().__init__()
        if len(y) != 0:
            self.dataset = TorchDataset(x, y)
        else:  # For pair-wise model, trained without given label
            self.dataset = PredictDataset(x)

    def generate_dataloader(self, x_test_user, x_all_item, batch_size, num_workers=8):
        train_dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = PredictDataset(x_test_user)

        # shuffle = False to keep same order as ground truth
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        item_dataset = PredictDataset(x_all_item)
        item_dataloader = DataLoader(item_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataloader, test_dataloader, item_dataloader

class DataGenerator(object):

    def __init__(self, x, y):
        super().__init__()
        self.dataset = TorchDataset(x, y)
        self.length = len(self.dataset)

    def generate_dataloader(self, x_val=None, y_val=None, x_test=None, y_test=None, split_ratio=None, batch_size=16,
                            num_workers=0):
        if split_ratio != None:
            train_length = int(self.length * split_ratio[0])
            val_length = int(self.length * split_ratio[1])
            test_length = self.length - train_length - val_length
            print("the samples of train : val : test are  %d : %d : %d" % (train_length, val_length, test_length))
            train_dataset, val_dataset, test_dataset = random_split(self.dataset,
                                                                    (train_length, val_length, test_length))
        else:
            train_dataset = self.dataset
            val_dataset = TorchDataset(x_val, y_val)
            test_dataset = TorchDataset(x_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataloader, val_dataloader, test_dataloader


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """ Pads sequences (list of list) to the ndarray of same length.
        This is an equivalent implementation of tf.keras.preprocessing.sequence.pad_sequences
        reference: https://github.com/huawei-noah/benchmark/tree/main/FuxiCTR/fuxictr
    
    Args:
        sequences (pd.DataFrame): data that needs to pad or truncate
        maxlen (int): maximum sequence length. Defaults to None.
        dtype (str, optional): Defaults to 'int32'.
        padding (str, optional): if len(sequences) less than maxlen, padding style, {'pre', 'post'}. Defaults to 'pre'.
        truncating (str, optional): if len(sequences) more than maxlen, truncate style, {'pre', 'post'}. Defaults to 'pre'.
        value (_type_, optional): Defaults to 0..

    Returns:
        _type_: _description_
    """
    

    assert padding in ["pre", "post"], "Invalid padding={}.".format(padding)
    assert truncating in ["pre", "post"], "Invalid truncating={}.".format(truncating)

    if maxlen is None:
        maxlen = max(len(x) for x in sequences)
    arr = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, x in enumerate(sequences):
        if len(x) == 0:
            continue  # empty list
        if truncating == 'pre':
            trunc = x[-maxlen:]
        else:
            trunc = x[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'pre':
            arr[idx, -len(trunc):] = trunc
        else:
            arr[idx, :len(trunc)] = trunc
    return arr

def df_to_dict(data):
    """
    Convert the DataFrame to a dict type input that the network can accept
    Args:
        data (pd.DataFrame): datasets of type DataFrame
    Returns:
        The converted dict, which can be used directly into the input network
    """
    data_dict = data.to_dict('list') # {name: list_list}
    for key in data.keys():
        data_dict[key] = np.array(data_dict[key])  # 转numpy
    return data_dict

def neg_sample(click_hist, item_size):
    neg = random.randint(1, item_size)
    while neg in click_hist:
        neg = random.randint(1, item_size)
    return neg


def array_replace_with_dict(array, dic):
    """Replace values in NumPy array based on dictionary.
    Args:
        array (np.array): a numpy array
        dic (dict): a map dict

    Returns:
        np.array: array with replace
    """
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    idx = k.argsort()
    return v[idx[np.searchsorted(k, array, sorter=idx)]]
