import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from math import ceil

class MLP(nn.Module):
    def __init__(self, data_dim, in_len):
        super(MLP, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        input_size=in_len*data_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_size, in_len),
            nn.ReLU(),
            nn.Linear(in_len, 1)
        )

    def forward(self, x):

        return self.mlp(x)


class Cross_MLP(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, baseline = False, device=torch.device('cuda:0')):
        super(Cross_MLP, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size

        self.baseline = baseline 

        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len
        

        self.MLP=MLP(data_dim, in_len)
    def forward(self, x_seq):
        
        x_seq = x_seq.flatten(1)
        predict_y = self.MLP(x_seq)

        return predict_y