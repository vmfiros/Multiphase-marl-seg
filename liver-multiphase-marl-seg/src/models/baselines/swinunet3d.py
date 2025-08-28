
import torch.nn as nn
from ..blocks.conv3d import ConvBlock3D
from ..blocks.swin3d import SwinBlock3D

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch)
        self.swin = SwinBlock3D(out_ch)
        self.pool = nn.Conv3d(out_ch, out_ch, 2, stride=2)
    def forward(self, x):
        x = self.conv(x); x = self.swin(x); skip = x; x = self.pool(x); return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock3D(out_ch*2, out_ch)
        self.swin = SwinBlock3D(out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        dz = skip.size(2)-x.size(2); dy = skip.size(3)-x.size(3); dx = skip.size(4)-x.size(4)
        x = nn.functional.pad(x, [0,dx,0,dy,0,dz])
        x = self.conv(nn.functional.relu(nn.functional.pad(x, [0,0,0,0,0,0]) if False else torch.cat([x, skip],1)))
        x = self.swin(x); return x

import torch
class SwinUNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, channels=(96,192,384), use_decoder_swin=True):
        super().__init__()
        c1,c2,c3 = channels
        self.d1 = Down(in_channels, c1)
        self.d2 = Down(c1, c2)
        self.d3 = Down(c2, c3)
        self.bottleneck = nn.Sequential(ConvBlock3D(c3, c3), SwinBlock3D(c3), SwinBlock3D(c3))
        self.u3 = Up(c3, c2)
        self.u2 = Up(c2, c1)
        self.u1 = Up(c1, c1)
        self.head = nn.Sequential(nn.Conv3d(c1, out_channels, 1), nn.Sigmoid())
    def forward(self, x):
        x,s1 = self.d1(x)
        x,s2 = self.d2(x)
        x,s3 = self.d3(x)
        x = self.bottleneck(x)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        p = self.head(x)
        return p, {}
