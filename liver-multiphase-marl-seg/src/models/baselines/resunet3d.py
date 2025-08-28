
import torch.nn as nn

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.down = nn.Conv3d(in_ch, out_ch, 1, stride=stride) if (in_ch!=out_ch or stride!=1) else nn.Identity()
    def forward(self, x):
        idn = self.down(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + idn)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ResBlock3D(in_ch, out_ch, stride=2)
    def forward(self, x):
        skip = x
        x = self.block(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.block = ResBlock3D(out_ch*2, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        dz = skip.size(2)-x.size(2); dy = skip.size(3)-x.size(3); dx = skip.size(4)-x.size(4)
        x = nn.functional.pad(x, [0,dx,0,dy,0,dz])
        x = self.block(nn.functional.pad(x, [0,0,0,0,0,0]) if False else nn.functional.relu(torch.cat([x, skip],1)))
        return x

import torch
class ResUNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_ch=32, depth=4):
        super().__init__()
        ch = [base_ch*(2**i) for i in range(depth)]
        self.stem = ResBlock3D(in_channels, ch[0], stride=1)
        self.downs = nn.ModuleList()
        prev = ch[0]
        for i in range(1, depth):
            self.downs.append(Down(prev, ch[i]))
            prev = ch[i]
        self.bottleneck = ResBlock3D(ch[-1], ch[-1])
        # decoder
        ups = []
        for i in range(depth-1, 0, -1):
            ups.append(Up(ch[i], ch[i-1]))
        self.ups = nn.ModuleList(ups)
        self.head = nn.Sequential(nn.Conv3d(ch[0], out_channels, 1), nn.Sigmoid())

    def forward(self, x):
        skips = []
        x = self.stem(x); skips.append(x)
        for d in self.downs:
            x, s = d(x); skips.append(s)
        x = self.bottleneck(x)
        for i,u in enumerate(self.ups):
            s = skips[-(i+2)]  # align
            x = u(x, s)
        p = self.head(x)
        return p, {}
