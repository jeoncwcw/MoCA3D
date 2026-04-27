from torch import nn
import torch

class DenseHeads(nn.Module):
    def __init__(self, heads, in_channels):
        super(DenseHeads, self).__init__()
        self.heads_dict = nn.ModuleDict()

        for role in heads:
            if role in ['corner heatmaps']:
                out_ch = 8  # 8 corners
                layers = [
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels, out_ch, kernel_size=1),
                    nn.Sigmoid()
                ]
            elif role in ['corner depths']:
                out_ch = 8 # Depth for each corner
                layers = [
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels, out_ch, kernel_size=1),
                    nn.Softplus(),
                ]
            else:
                raise ValueError(f"Unknown head role: {role}")
            
            fc = nn.Sequential(*layers)
            self._init_weights(fc)
            
            if "heatmap" in role:
                fc[-2].bias.data.fill_(-2.19)
                

            self.heads_dict[role] = fc

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        outputs = {}
        for role, head in self.heads_dict.items():
            outputs[role] = head(x)
        return outputs