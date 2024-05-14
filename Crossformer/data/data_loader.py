import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, 
                  data_split = [0.7, 0.1, 0.2], scale=False, scale_statistic=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.in_len = size[0]
        self.out_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.eps=1e-5
        
        #self.inverse = inverse
        
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if (self.data_split[0] > 1):
            train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
        else:
            train_num = int(len(df_raw)*self.data_split[0]); 
            test_num = int(len(df_raw)*self.data_split[2])
            val_num = len(df_raw) - train_num - test_num; 
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        
        data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_y = self.data_y[:, -1:]

    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.in_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x = self._normalize(seq_x)
        seq_y = self.data_y[s_end-1:s_end]
        #这里要找的标签实际上就是x最后一行对应的logreturn

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.in_len- self.out_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def _normalize(self, x):
        #x: [timestamp, features], normlize along the timestamp
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_std = np.std(x, axis=0, keepdims=True) + self.eps
        x_normed = (x - x_mean) / x_std

        # x=torch.from_numpy(x)
        # dim2reduce = tuple(range(0, x.ndim-1))
        # self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        # self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        # x = x - self.mean
        # x = x / self.stdev
        # x=x.numpy()

        return x_normed