import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path='none', flag='train', size=None, 
                  data_split = [0.7, 0.1, 0.2], scale_factor=100):
        # size [seq_len, label_len, pred_len]
        # info
        self.in_len = size[0]
        self.out_len = 1
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.eps=1e-5

        self.scale_factor = scale_factor
        
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.data_Xs=[]
        self.data_Ys=[]                     
        self.instance_nums=[]
        self.__read_data__()

    def __read_data__(self):
        #read all datanames in the root_path
        data_names = os.listdir(self.root_path)

        train_num=0
        test_num=0
        val_num=0
        for name in data_names:
            df_raw = pd.read_csv(os.path.join(self.root_path, name))
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
            data_x = data[border1:border2, :-1]
            data_y = data[border1:border2, -1:]
            self.data_Xs.append(data_x)
            self.data_Ys.append(data_y)
            instance_num = border2 - border1 - self.in_len + 1
            self.instance_nums.append(instance_num)
        self.instance_nums = np.array(self.instance_nums)
        self.prefix_sum = np.cumsum(self.instance_nums)
    
    def __getitem__(self, index):  
        # 根据self.nums确定index对应的子集i和数据在该子集内的偏移量  

        dataset_idx = np.searchsorted(self.prefix_sum, index, side='right')
        instance_idx = index if dataset_idx == 0 else index - self.prefix_sum[dataset_idx - 1]
        s_begin = instance_idx
        s_end = s_begin + self.in_len
        
        seq_x = self.data_Xs[dataset_idx][s_begin:s_end]  
        seq_x = self._normalize(seq_x)  # 数据标准化  
        seq_y = self.data_Ys[dataset_idx][s_begin + self.in_len - 1]  
      
        return seq_x, seq_y * self.scale_factor
  
    def __len__(self):  
        return self.prefix_sum[-1]

    def _normalize(self, x):
        #x: [timestamp, features], normlize along the timestamp
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_std = np.std(x, axis=0, keepdims=True) + self.eps
        x_normed = (x - x_mean) / x_std

        return x_normed