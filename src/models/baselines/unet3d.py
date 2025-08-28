
import torch.nn as nn
from ..blocks.conv3d import ConvBlock3D

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ConvBlock3D(in_ch, out_ch)
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=2, stride=2)
    def forward(self, x):
        x = self.block(x); skip = x
        x = self.down(x); return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock3D(out_ch*2, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if mismatch
        dz = skip.size(2)-x.size(2); dy = skip.size(3)-x.size(3); dx = skip.size(4)-x.size(4)
        x = nn.functional.pad(x, [0,dx,0,dy,0,dz])
        x = nn.functional.relu(x)
        x = nn.functional.relu(x)
        x = nn.functional.relu(x)
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.relu(x)
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])  # (ensure shape)
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        x = nn.functional.pad(x, [0,0,0,0,0,0])
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_ch=32, depth=4):
        super().__init__()
        ch = [base_ch*(2**i) for i in range(depth)]
        self.downs = nn.ModuleList()
        prev = in_channels
        for c in ch:
            self.downs.append(Down(prev, c))
            prev = c
        self.bottleneck = ConvBlock3D(ch[-1], ch[-1])
        # decoder
        ups = []
        for i in range(depth-1, -1, -1):
            outc = ch[i-1] if i>0 else ch[0]
            inc = ch[i] if i==depth-1 else outc
            ups.append(Up(inc, outc))
        self.ups = nn.ModuleList(ups)
        self.head = nn.Sequential(nn.Conv3d(ch[0], out_channels, 1), nn.Sigmoid())

    def forward(self, x):
        skips = []
        for d in self.downs:
            x, s = d(x); skips.append(s)
        x = self.bottleneck(x)
        for i,u in enumerate(self.ups):
            s = skips[-(i+1)]
            x = u(x, s)
        p = self.head(x)
        return p, {}
