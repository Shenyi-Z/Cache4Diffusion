# Besides, re-arrange the attention module
from torch.jit import Final
from timm.layers import use_fused_attn
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        #self.fused_attn = False
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, cache_dic, current, fresh_indices=None) -> torch.Tensor:
    # 0.4ms extra cost on A800, mainly tensor operations
        """
        fresh_indices: (B, fresh_ratio*N), the index tensor for the fresh tokens
        """

        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   #q: (B, num_heads, N, head_dim)

        q, k = self.q_norm(q), self.k_norm(k)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            attn= attn.softmax(dim=-1) 
            attn = self.attn_drop(attn)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x) 
        
        flops = (
            B * N * C * 3 * C * 2 # QKV projection
            + B * self.num_heads * N * self.head_dim  # Scale q
            + B * self.num_heads * N * N * self.head_dim * 2 # Q @ K
            + B * self.num_heads * N * N * 5 # Softmax
            + B * self.num_heads * N * N * self.head_dim * 2 # Attn @ V
            + B * N * C * C * 2 # Projection
        )
        cache_dic['flops']+=flops
        return x
