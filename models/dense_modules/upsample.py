from torch import nn
import torch

class UpsampleLayer(nn.Module):
    def __init__(self, d_model, skip_channels=None, activation="relu"):
        super(UpsampleLayer, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(d_model + skip_channels, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            _get_activation_fn(activation),
            ResBlock(d_model, activation=activation),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            _get_activation_fn(activation),
            ResBlock(d_model // 2, activation=activation),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(d_model // 2, d_model // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 4),
            _get_activation_fn(activation),
            ResBlock(d_model // 4, activation=activation),
        )
    def forward(self, x, skip_feature=None):
        if skip_feature is not None:
            x = torch.cat([x, skip_feature], dim=1)
            x = self.fusion(x)
        x = self.up1(x)
        x = self.up2(x)
        return x

# class LayerNorm2d(nn.Module):
#     def __init__(self, num_features, eps=1e-6):
#         super(LayerNorm2d, self).__init__()
#         self.weight = nn.Parameter(torch.ones(num_features))
#         self.bias = nn.Parameter(torch.zeros(num_features))
#         self.eps = eps

#     def forward(self, x):
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)

#         return self.weight[:, None, None] * x + self.bias[:, None, None]
    
def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

class ResBlock(nn.Module):
    def __init__(self, channels, activation="relu"):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            _get_activation_fn(activation),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = _get_activation_fn(activation)
    def forward(self, x):
        return self.activation(x + self.block(x))
        