import torch
import torch.nn as nn
import torch.nn.functional as F
from ..blocks.conv3d import ConvBlock3D

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ConvBlock3D(in_ch, out_ch)            # conv-norm-act x2
        self.down = nn.Conv3d(out_ch, out_ch, 2, stride=2) # ↓/2
    def forward(self, x):
        x = self.block(x)
        skip = x
        x = self.down(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)  # ↑×2
        self.conv = ConvBlock3D(out_ch * 2, out_ch)               # concat → conv
    def forward(self, x, skip):
        x = self.up(x)
        # pad to match skip spatial dims
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        if dz or dy or dx:
            x = F.pad(x, [0, dx, 0, dy, 0, dz])
        x = torch.cat([x, skip], dim=1)   # ⊕ skip
        x = self.conv(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_ch=32, depth=4):
        super().__init__()
        assert depth >= 2, "depth must be ≥2"
        ch = [base_ch * (2 ** i) for i in range(depth)]  # e.g., [32,64,128,256]

        # encoder
        self.downs = nn.ModuleList()
        prev = in_channels
        for c in ch:
            self.downs.append(Down(prev, c))
            prev = c

        # bottleneck (no further downsample)
        self.bottleneck = ConvBlock3D(ch[-1], ch[-1])

        # decoder: number of ups = number of downs
        ups = []
        # First up takes bottleneck (ch[-1]) → ch[-2]
        ups.append(Up(ch[-1], ch[-2]))
        # Middle ups
        for i in range(depth - 2, 0, -1):   # e.g., i = 1 for depth=3
            ups.append(Up(ch[i], ch[i - 1]))
        # Last up restores to top level (ch[0])
        ups.append(Up(ch[0], ch[0]))
        self.ups = nn.ModuleList(ups)

        self.head = nn.Sequential(nn.Conv3d(ch[0], out_channels, 1), nn.Sigmoid())

    def forward(self, x):
        skips = []
        for d in self.downs:
            x, s = d(x)
            skips.append(s)
        x = self.bottleneck(x)

        # use skips in reverse order
        for i, u in enumerate(self.ups):
            s = skips[-(i + 1)]
            x = u(x, s)

        p = self.head(x)
        return p, {}
