
# A lightweight 'Swin-like' 3D attention block using windowed MHSA.
# For journal reproducibility without pulling heavy external code.
import torch, torch.nn as nn
from einops import rearrange

class WindowAttention3D(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=(4,4,4)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ws = window_size
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B,C,D,H,W = x.shape
        wd,wh,ww = self.ws
        # pad to multiple of window
        pd = (wd - D%wd) % wd
        ph = (wh - H%wh) % wh
        pw = (ww - W%ww) % ww
        x = nn.functional.pad(x, (0,pw,0,ph,0,pd))
        B,C,Dp,Hp,Wp = x.shape
        # partition windows and flatten (tokens)
        xw = rearrange(x, "b c (nd wd) (nh wh) (nw ww) -> (b nd nh nw) (wd wh ww) c", wd=wd, wh=wh, ww=ww)
        # attend within window
        out, _ = self.attn(xw, xw, xw)  # (num_win, tokens, C)
        out = self.proj(out)
        # reverse windows
        out = rearrange(out, "(b nd nh nw) (wd wh ww) c -> b c (nd wd) (nh wh) (nw ww)",
                        b=B, nd=Dp//wd, nh=Hp//wh, nw=Wp//ww, wd=wd, wh=wh, ww=ww, c=C)
        # remove pad
        out = out[:, :, :D, :H, :W]
        return out

class SwinBlock3D(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=(4,4,4)):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(dim, affine=True)
        self.attn = WindowAttention3D(dim, num_heads, window_size)
        self.norm2 = nn.InstanceNorm3d(dim, affine=True)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, dim*4, 1),
            nn.GELU(),
            nn.Conv3d(dim*4, dim, 1),
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
