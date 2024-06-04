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
    

class Dataset_multiStock(Dataset):
    def __init__(self, root_path, data_path='data', timestamp='timestamps.npy', flag='train', in_len=96, 
                  data_split = [0.7, 0.1, 0.2], scale_factor=100):
        # info
        self.in_len = in_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.eps=1e-5

        self.scale_factor = scale_factor
        
        self.root_path = root_path
        self.data_path = data_path
        self.timestamp = timestamp
        self.data_split = data_split
        self.stock_ids = []
        self.timestamps = []
        self.data_Xs=[]
        self.data_Ys=[]                     
        self.instance_nums=[]
        self.__read_data__()

    def __read_data__(self):
        #read timestamps and choose the spliting timepoints
        timestamps = np.load(os.path.join(self.root_path, self.timestamp))
        train_num_tmp = int(len(timestamps)*self.data_split[0])
        test_num_tmp = int(len(timestamps)*self.data_split[2])
        val_num_tmp = len(timestamps) - train_num_tmp - test_num_tmp
        train_val_sep = timestamps[train_num_tmp]
        val_test_sep = timestamps[train_num_tmp + val_num_tmp]
        if self.set_type == 0:
            print('Data points earlier than {}(not included) are for training'.format(train_val_sep))
            print('Data points from {}(included) to {}(not included) are for validation'.format(train_val_sep, val_test_sep))
            print('Data points from {}(included) are for testing'.format(val_test_sep))

        #read all datanames in the root_path
        data_names = os.listdir(os.path.join(self.root_path, self.data_path))

        for name in data_names:
            stock_id = int(name.split('.')[0].split('S')[-1])
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path, name))

            #split the data
            df_raw['time'] = pd.to_datetime(df_raw['time'])
            train_num = df_raw[df_raw['time'] < train_val_sep].shape[0]
            test_num = df_raw[df_raw['time'] >= val_test_sep].shape[0]
            val_num = len(df_raw) - train_num - test_num
            border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
            border2s = [train_num, train_num+val_num, train_num + val_num + test_num]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            data = df_data.values
            data_x = data[border1:border2, :-1]
            data_y = data[border1:border2, -1:]
            #check if there is nan in data_y
            if np.isnan(data_y.min()):
                print('data_y.min() is nan')
                print(name)
            if np.isinf(data_y.min()):
                print('data_y.min() is inf')
                print(name)
            self.data_Xs.append(data_x)
            self.data_Ys.append(data_y)
            instance_num = border2 - border1 - self.in_len + 1
            self.instance_nums.append(instance_num)

            self.stock_ids.append(stock_id*torch.ones(instance_num))
            times = df_raw['time'].values
            self.timestamps.append(times[border1:border2])

        self.instance_nums = np.array(self.instance_nums)
        self.prefix_sum = np.cumsum(self.instance_nums)
    
    def __getitem__(self, index):  
        # 根据prefix_sum确定index对应的子集i和数据在该子集内的偏移量  
        dataset_idx = np.searchsorted(self.prefix_sum, index, side='right')
        instance_idx = index if dataset_idx == 0 else index - self.prefix_sum[dataset_idx - 1]
        s_begin = instance_idx
        s_end = s_begin + self.in_len
        
        seq_x = self.data_Xs[dataset_idx][s_begin:s_end]  
        seq_x = self._normalize(seq_x)  # 数据标准化  
        #check if seq_x.min() is nan
        seq_y = self.data_Ys[dataset_idx][s_end - 1]  
        if np.isnan(seq_y.min()):
            print('seq_y.min() is nan')
        if np.isinf(seq_y.min()):
            print('seq_y.min() is inf')
        timestamp = self.timestamps[dataset_idx][s_end - 1]
        stock_id = self.stock_ids[dataset_idx][instance_idx]

        timestamp_tmp = split_timedata(timestamp)
      
        return seq_x, seq_y * self.scale_factor, timestamp_tmp, stock_id
  
    def __len__(self):  
        return self.prefix_sum[-1]

    def _normalize(self, x):
        #x: [timestamp, features], normlize along the timestamp
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_std = np.std(x, axis=0, keepdims=True) + self.eps
        x_normed = (x - x_mean) / x_std

        return x_normed

def split_timedata(timestamp):
    #split a dattime64 timestamp into year, month, day, hour, minute
    timestamp = pd.to_datetime(timestamp)
    year = timestamp.year
    month = timestamp.month
    day = timestamp.day
    hour = timestamp.hour
    minute = timestamp.minute

    time_array = np.array([year, month, day, hour, minute])
    return time_array

def merge_timearray(time_array):
    #merge a time array into a datetime64 timestamp
    year = time_array[0]
    month = time_array[1]
    day = time_array[2]
    hour = time_array[3]
    minute = time_array[4]

    timestamp = pd.to_datetime('{}-{}-{} {}:{}:00'.format(year, month, day, hour, minute))
    return timestamp
