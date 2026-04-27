import torch 
import torch.nn as nn
from .utils import _get_clones, _get_activation_fn
from typing import Optional

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                image_feat,
                image_feat_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None):
        output = image_feat

        for layer in self.layers:
            output = layer(output, image_feat_key_padding_mask=image_feat_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=True):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, image_feat, image_feat_key_padding_mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(image_feat, image_feat_key_padding_mask, pos)
        return self.forward_post(image_feat, image_feat_key_padding_mask, pos)
    
    def forward_post(self,
                     image_feat,
                     image_feat_key_padding_mask: Optional[torch.Tensor] = None,
                     pos: Optional[torch.Tensor] = None):
        q = k = self.with_pos_embed(image_feat, pos)
        image_feat2 = self.self_attn(q, k, value=image_feat, key_padding_mask=image_feat_key_padding_mask)[0]
        image_feat = image_feat + self.dropout1(image_feat2)
        image_feat = self.norm1(image_feat)
        image_feat2 = self.linear2(self.dropout(self.activation(self.linear1(image_feat))))
        image_feat = image_feat + self.dropout2(image_feat2)
        image_feat = self.norm2(image_feat)
        return image_feat
    
    def forward_pre(self,
                    image_feat,
                    image_feat_key_padding_mask: Optional[torch.Tensor] = None,
                    pos: Optional[torch.Tensor] = None):
        image_feat2 = self.norm1(image_feat)
        q = k = self.with_pos_embed(image_feat2, pos)
        image_feat2 = self.self_attn(q, k, value=image_feat2, key_padding_mask=image_feat_key_padding_mask)[0]
        image_feat = image_feat + self.dropout1(image_feat2)
        image_feat2 = self.norm2(image_feat)
        image_feat2 = self.linear2(self.dropout(self.activation(self.linear1(image_feat2))))
        image_feat = image_feat + self.dropout2(image_feat2)
        return image_feat
    
