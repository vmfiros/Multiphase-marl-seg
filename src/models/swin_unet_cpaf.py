
import torch, torch.nn as nn
from .blocks.conv3d import ConvBlock3D
from .blocks.swin3d import SwinBlock3D
from .blocks.cpaf import CPAF

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_swin=True):
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch)
        self.swin = SwinBlock3D(out_ch) if use_swin else nn.Identity()
        self.pool = nn.Conv3d(out_ch, out_ch, 2, stride=2)  # downsample
    def forward(self, x):
        x = self.conv(x)
        x = self.swin(x)
        skip = x
        x = self.pool(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, use_swin=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock3D(out_ch*2, out_ch)
        self.swin = SwinBlock3D(out_ch) if use_swin else nn.Identity()
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)
        x = nn.functional.pad(x, [0, diffX, 0, diffY, 0, diffZ])
        x = torch.cat([x, skip], dim=1)  # ⊕
        x = self.conv(x)
        x = self.swin(x)
        return x

class SwinUNetCPAF(nn.Module):
    """Dual encoders (arterial/venous) with CPAF at L1–L3, bottleneck, and decoder."""
    def __init__(self, in_channels=2, out_channels=1, channels=(96,192,384), use_decoder_swin=True):
        super().__init__()
        assert in_channels==2, "Expect 2-channel input (arterial, venous)."
        c1,c2,c3 = channels
        # Split 2-channel input into two streams
        self.split = lambda x: (x[:,0:1], x[:,1:2])  # (B,1,D,H,W) each

        # Encoders (A & V) share the same structure but separate params
        self.A1 = Down(1, c1); self.V1 = Down(1, c1)
        self.cpaf1 = CPAF(c1)
        self.A2 = Down(c1, c2); self.V2 = Down(c1, c2)
        self.cpaf2 = CPAF(c2)
        self.A3 = Down(c2, c3); self.V3 = Down(c2, c3)
        self.cpaf3 = CPAF(c3)

        # Bottleneck
        self.bottleneck = nn.Sequential(ConvBlock3D(c3, c3), SwinBlock3D(c3), SwinBlock3D(c3))

        # Decoder (single stream on fused skips)
        self.up3 = Up(c3, c2, use_swin=use_decoder_swin)
        self.up2 = Up(c2, c1, use_swin=use_decoder_swin)
        self.up1 = Up(c1, c1, use_swin=use_decoder_swin)

        self.head = nn.Sequential(nn.Conv3d(c1, out_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        # split
        xa, xv = self.split(x)
        # L1
        xa, sa = self.A1(xa); xv, sv = self.V1(xv)
        s1 = self.cpaf1(sa, sv)
        # L2
        xa, sa = self.A2(xa); xv, sv = self.V2(xv)
        s2 = self.cpaf2(sa, sv)
        # L3
        xa, sa = self.A3(xa); xv, sv = self.V3(xv)
        s3 = self.cpaf3(sa, sv)
        # bottleneck (use last encoder x from one stream; both are same spatial size)
        x = self.bottleneck(sa)  # use arterial deepest; could also fuse sa,sv

        # decoder with fused skips
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        p = self.head(x)  # probability map
        return p, {'skip1': s1, 'skip2': s2, 'skip3': s3, 'bottleneck': x}
