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
        self.out_len = 1
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.eps=1e-5
        
        #self.inverse = inverse
        
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.data_Xs=[]
        self.data_Ys=[]                     
        self.nums=[]
        self.datapath=['S100.csv','S200.csv','S400.csv','S600.csv']
        self.i=0
        self.__read_data__()

    def __read_data__(self):
        train_num=0
        test_num=0
        val_num=0
        for i in [0,1,2,3]:
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.datapath[i]))
            train_num = int(len(df_raw)*self.data_split[0]); 
            test_num = int(len(df_raw)*self.data_split[2])
            val_num = len(df_raw) - train_num - test_num; 
            border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
            border2s = [train_num, train_num+val_num, train_num + val_num + test_num]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            cols_data = df_raw.columns[1:]
            df_datax = df_raw[cols_data[:-1]]
            df_data = df_raw[cols_data]
            data = df_data.values
            datax = df_datax.values
            self.data_x = datax[border1:border2]
            self.data_y = data[border1:border2]
            self.data_y = self.data_y[:, -1:]
            self.data_Xs.append(self.data_x)
            self.data_Ys.append(self.data_y)
            l=border2-border1
            self.nums.append(l)
 



    
    def __getitem__(self, index):  
        # 根据self.nums确定index对应的子集i和数据在该子集内的偏移量  
        t = 0  
        i = 0  
        while t + self.nums[i] <index+self.in_len*(i+1):  
            t += self.nums[i]  
            i += 1  
            if i==len(self.nums)-1:
                break
        #print(-1)
        #print(index)
        #print(t)
        #print(i)
        local_index=index
        if (i>0):
            local_index=index-(t-self.in_len*i)-1
        #print(local_index)
        s_begin = local_index  
        s_end = s_begin + self.in_len  
        
        seq_x = self.data_Xs[i][s_begin:s_end]  
        #print(seq_x.shape)
        seq_x = self._normalize(seq_x)  # 数据标准化  
        seq_y = self.data_Ys[i][s_begin + self.in_len - 1:s_begin + self.in_len]  
        seq_y = seq_y.squeeze(-1)  


      
        return seq_x, seq_y  
  
    def __len__(self):  
        total_length = 0  
        for num in self.nums:  
            total_length += num - self.in_len + 1  
        return total_length+1-len(self.nums)

    def _normalize(self, x):
        #x: [timestamp, features], normlize along the timestamp
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_std = np.std(x, axis=0, keepdims=True) + self.eps
        x_std[x_std == 0] = 1e-8
        x_normed = (x - x_mean) / x_std

        # x=torch.from_numpy(x)
        # dim2reduce = tuple(range(0, x.ndim-1))
        # self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        # self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        # x = x - self.mean
        # x = x / self.stdev
        # x=x.numpy()

        return x_normed