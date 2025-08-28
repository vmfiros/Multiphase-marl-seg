
# Cross-Phase Attention Fusion (arterial↔venous) at each encoder level
import torch, torch.nn as nn
from einops import rearrange

class CPAF(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.qkv_a = nn.Linear(dim, dim*3)
        self.qkv_v = nn.Linear(dim, dim*3)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim*2, dim)  # fuse A' and V'
    def forward(self, fa, fv):
        # fa,fv: (B, C, D, H, W)
        B,C,D,H,W = fa.shape
        # flatten spatial
        ta = rearrange(fa, "b c d h w -> b (d h w) c")
        tv = rearrange(fv, "b c d h w -> b (d h w) c")
        # A attends to V
        out_a, _ = self.attn(ta, tv, tv)
        # V attends to A
        out_v, _ = self.attn(tv, ta, ta)
        fused = torch.cat([out_a, out_v], dim=-1)  # (B, N, 2C)
        fused = self.proj(fused)                   # (B, N, C)
        fused = rearrange(fused, "b n c -> b c d h w", d=D, h=H, w=W)
        return fused
