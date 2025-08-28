import torch
import torch.nn as nn
import torch.nn.functional as F
from ..blocks.conv3d import ConvBlock3D

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ConvBlock3D(in_ch, out_ch)                 # conv-norm-act x2
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=2, stride=2)  # ↓/2
    def forward(self, x):
        x = self.block(x)
        skip = x                                               # keep for decoder
        x = self.down(x)
        return x, skip

class Up(nn.Module):
    """
    Upsample from in_ch to skip_ch, then concat with skip (skip_ch) → ConvBlock3D(2*skip_ch → skip_ch)
    """
    def __init__(self, in_ch, skip_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, skip_ch, kernel_size=2, stride=2)  # ↑×2 to match skip spatial size
        self.conv = ConvBlock3D(skip_ch + skip_ch, skip_ch)                    # concat channels → reduce to skip_ch
    def forward(self, x, skip):
        x = self.up(x)
        # pad if shapes off by 1 voxel due to pooling/transpose rounding
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        if dz or dy or dx:
            x = F.pad(x, [0, dx, 0, dy, 0, dz])
        x = torch.cat([x, skip], dim=1)  # ⊕ skip
        x = self.conv(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_ch=32, depth=3):
        """
        depth=3 → channels [32, 64, 128] by default
        """
        super().__init__()
        assert depth >= 2, "depth must be ≥2"
        ch = [base_ch * (2 ** i) for i in range(depth)]  # e.g., [32, 64, 128]

        # encoder
        self.downs = nn.ModuleList()
        prev = in_channels
        for c in ch:
            self.downs.append(Down(prev, c))
            prev = c

        # bottleneck
        self.bottleneck = ConvBlock3D(ch[-1], ch[-1])

        # decoder: Up(in_ch, skip_ch) where skip_ch == ch[i] from encoder
        ups = []
        # first up: bottleneck (ch[-1]) with deepest skip (ch[-1])
        ups.append(Up(ch[-1], ch[-1]))          # output channels = ch[-1]
        # then step up through remaining skips
        for i in range(depth - 2, -1, -1):      # i = depth-2 ... 0
            ups.append(Up(ch[i + 1] if i + 1 < len(ch) else ch[i], ch[i]))
        self.ups = nn.ModuleList(ups)

        self.head = nn.Sequential(nn.Conv3d(ch[0], out_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        skips = []
        for d in self.downs:
            x, s = d(x)
            skips.append(s)                     # s has channels ch[i]

        x = self.bottleneck(x)                  # channels ch[-1]

        # use skips in reverse order
        # first Up uses deepest skip (skips[-1]), already sized for ch[-1]
        x = self.ups[0](x, skips[-1])
        # next Ups use the remaining skips: skips[-2], ..., skips[0]
        for i, u in enumerate(self.ups[1:], start=1):
            x = u(x, skips[-(i + 1)])

        p = self.head(x)                        # channels → 1
        return p, {}
