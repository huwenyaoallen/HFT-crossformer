import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from cross_models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer

class DecoderLayer(nn.Module):
    '''
    The decoder layer of Crossformer, each layer will make a prediction at its scale
    '''
    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num = 10, factor = 10):
        super(DecoderLayer, self).__init__()
        #存储自注意力和交叉注意力趋势
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, \
                                d_ff, dropout)    
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
        #定义两个正则化层，用于标准化数据
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        #多层感知机

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.GELU(),
                                nn.Linear(d_model, d_model))
        #映射到段的长度
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        '''
        x: the output of last decoder layer
        cross: the output of the corresponding encoder layer
        '''
        #批次大小
        batch = x.shape[0]
        x = self.self_attention(x)
        #重组从而匹配交叉注意力
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        
        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        tmp = self.cross_attention(
            x, cross, cross,
        )
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x+y)
        
        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b = batch)
        layer_predict = self.linear_pred(dec_output)

        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')
    
        print(layer_predict.shape)
        return dec_output, layer_predict
class MaxPoolingLayer(nn.Module):
    def __init__(self, d_model):
        super(MaxPoolingLayer, self).__init__()
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.GELU(),
                                nn.Linear(d_model, d_model))
        #映射到段的长度
        self.linear_pred = nn.Linear(d_model, 1)

    def forward(self, x):
        # 对 seg_num 维度进行最大池化操作
        pooled_output, _ = torch.max(x, dim=2, keepdim=True)
        # 对 ts_d 维度进行平均池化操作
        pooled_output,_ = torch.max(pooled_output, dim=1,keepdim=True)
        pooled_output = self.MLP1(pooled_output)

        layer_predict = self.linear_pred(pooled_output)
        #print(layer_predict.shape)


        layer_predict = rearrange(layer_predict, 'b 1 1 1 -> b 1')
        #print(layer_predict.shape)

        return layer_predict
#x输入:b ts_d seg_num d_model
class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout,\
                router=False, out_seg_num = 10, factor=10):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(MaxPoolingLayer(d_model))
    #接受输出为x
    def forward(self, x, cross):
        #提取时间步长维度
        final_predict = None
        i = 0
        #获取时间步长
        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            layer_predict = layer(cross_enc)
            #第一次迭代，预测为该层预测，否则进行累加
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        return final_predict