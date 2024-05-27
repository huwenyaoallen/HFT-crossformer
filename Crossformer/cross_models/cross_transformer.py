import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torch import tensor
import torch.utils.data as Data
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import math
from datetime import datetime, timedelta
from math import ceil

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

 
    def forward(self, x: torch.Tensor):
        chunk = x.chunk(x.size(-1), dim=2)
        out = torch.Tensor([]).to(x.device)
        for i in range(len(chunk)):
            out = torch.cat((out, chunk[i] + self.pe[:chunk[i].size(0), ...]), dim=2)
        return out

def transformer_generate_tgt_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1
    mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
    )
    return mask

class Transformer(nn.Module):

    def __init__(self, data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, baseline = False, device=torch.device('cuda:0')):
        super(Transformer, self).__init__()
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
 
        self.input_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)
        self.target_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)
 
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout,
                                                         dim_feedforward=4 * d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dropout=dropout,
                                                         dim_feedforward=4 * d_model)
 
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=4)
 
        self.input_projection = torch.nn.Linear(data_dim, d_model)
        self.output_projection = torch.nn.Linear(data_dim, d_model)
        self.linear = torch.nn.Linear(d_model, 1)
        self.linear1=torch.nn.Linear(in_len, 1)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.GELU(),
                                nn.Linear(d_model, d_model))

 
    

    def encode_in(self, src):
        src_start = self.input_projection(src).permute(1, 0, 2)
        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (torch.arange(0, in_sequence_len, device=src.device).unsqueeze(0).repeat(batch_size, 1))
        #print(2)
        #print(pos_encoder.shape)
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)
        #print(pos_encoder.shape)
        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        return src

 
    def forward(self, src):
        src = self.encode_in(src)
        src = self.MLP1(src)
        pooled_output,_ = torch.max(src, dim=2)
        pooled_output,_ = torch.max(pooled_output, dim=0)
        predict_y=pooled_output
        return predict_y

                                           


