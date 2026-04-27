import torch
import torch.nn as nn
from .utils import _get_clones, _get_activation_fn
from typing import Optional

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, image_feat, box_embed,
                image_feat_key_padding_mask: Optional[torch.Tensor] = None,
                image_feat_pos: Optional[torch.Tensor] = None):
        output = image_feat

        for layer in self.layers:
            output = layer(
                output,
                box_embed,
                image_feat_key_padding_mask=image_feat_key_padding_mask,
                image_feat_pos=image_feat_pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=True):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, image_feat, box_embed,
                image_feat_key_padding_mask: Optional[torch.Tensor] = None,
                image_feat_pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(image_feat, box_embed, image_feat_key_padding_mask, image_feat_pos)
        return self.forward_post(image_feat, box_embed, image_feat_key_padding_mask, image_feat_pos)
    
    def forward_post(self, image_feat, box_embed,
                     image_feat_key_padding_mask: Optional[torch.Tensor] = None,
                     image_feat_pos: Optional[torch.Tensor] = None):
        
        tgt = self.with_pos_embed(image_feat, image_feat_pos)
        tgt2 = self.self_attn(tgt, tgt, value=image_feat, key_padding_mask=image_feat_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(query=self.with_pos_embed(tgt, image_feat_pos),
                               key = box_embed,
                               value = box_embed)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward_pre(self, image_feat, box_embed,
                    image_feat_key_padding_mask: Optional[torch.Tensor] = None,
                    image_feat_pos: Optional[torch.Tensor] = None):
        
        tgt = image_feat
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, image_feat_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, key_padding_mask=image_feat_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(query=self.with_pos_embed(tgt2, image_feat_pos),
                               key = box_embed,
                               value = box_embed)[0]
        tgt = tgt + self.dropout2(tgt2)
        
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt