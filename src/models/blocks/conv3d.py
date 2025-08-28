
import torch.nn as nn

def conv3x3x3(in_ch, out_ch, stride=1):
    return nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm='bn'):
        super().__init__()
        Norm = nn.BatchNorm3d if norm=='bn' else nn.InstanceNorm3d
        self.block = nn.Sequential(
            conv3x3x3(in_ch, out_ch, stride),
            Norm(out_ch),
            nn.GELU(),
            conv3x3x3(out_ch, out_ch, 1),
            Norm(out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)
