import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from cross_models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer


class DecoderLayer(nn.Module):
    '''
    The decoder layer of Crossformer, each layer will make a prediction at its scale
    '''
    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num=10, factor=10):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x):
        '''
        x: the output of last decoder layer
        '''
        x = self.self_attention(x)
        x = self.norm1(x)
        x = self.MLP1(x)
        dec_output = self.linear_pred(x)
        return dec_output


class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout, router=False, out_seg_num=10, factor=10):
        super(Decoder, self).__init__()
        self.router = router
        self.decode_layers = nn.ModuleList()
        self.max_pool = nn.MaxPool1d(seg_len)  # Max Pooling layer
        self.mlp_layers = nn.ModuleList()
        self.num_scales = d_layers
        self.weight_layer = nn.Linear(d_model, d_layers)  # Weighting layer for different scales
        for _ in range(d_layers):
            mlp_layer = nn.Sequential(nn.Linear(d_model, d_model),
                                      nn.GELU(),
                                      nn.Linear(d_model, d_model))
            self.mlp_layers.append(mlp_layer)

    def forward(self, x):
        final_predict = None

        for i, layer in enumerate(self.decode_layers):
            layer_output = layer(x)  # Apply DecoderLayer to get predictions at each scale
            mlp_output = self.mlp_layers[i](layer_output)  # Apply MLP specific to the scale
            if final_predict is None:
                final_predict = mlp_output
            else:
                final_predict += mlp_output

        final_predict = final_predict / self.num_scales  # Average the predictions across scales

        # Apply Max Pooling along the time axis
        pooled_output = self.max_pool(final_predict.permute(0, 2, 1))
        pooled_output = pooled_output.squeeze(2)

        # Apply weighting layer for different scales
        weights = self.weight_layer(pooled_output)
        weights = torch.softmax(weights, dim=1)

        # Weighted sum of predictions from different scales
        weighted_predictions = final_predict * weights.unsqueeze(2)
        final_predict = torch.sum(weighted_predictions, dim=1)

        return final_predict