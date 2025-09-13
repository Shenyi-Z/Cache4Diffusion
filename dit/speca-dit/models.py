# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import os
import torch
import torch.nn as nn
import numpy as np
import math
#from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.models.vision_transformer import PatchEmbed, Mlp
#import os.path as osp
from cache_functions import Attention, cal_type
from taylor_utils import derivative_approximation, taylor_formula, taylor_cache_init

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                           误差计算函数                                        #
#################################################################################

def calculate_l1_error(x, full_x):
    """
    计算L1误差 (平均绝对误差)
    """
    return torch.abs(x - full_x).mean().item()

def calculate_l2_error(x, full_x):
    """
    计算L2误差 (均方根误差)
    """
    return torch.sqrt(torch.mean((x - full_x) ** 2)).item()

def calculate_relative_l1_error(x, full_x, eps=1e-10):
    """
    计算相对L1误差
    """
    error = torch.abs(x - full_x) / (torch.abs(full_x) + eps)
    return error.mean().item()

def calculate_relative_l2_error(x, full_x, eps=1e-10):
    """
    计算相对L2误差
    """
    error = torch.abs(x - full_x) / (torch.abs(full_x) + eps)
    return torch.sqrt(torch.mean(error ** 2)).item()

def calculate_cosine_similarity_error(x, full_x, eps=1e-10):
    """
    计算cosine相似度误差 (1 - cosine_similarity)
    """
    # 将张量展平为向量
    x_flat = x.view(x.size(0), -1)
    full_x_flat = full_x.view(full_x.size(0), -1)
    
    # 计算cosine相似度
    cosine_sim = torch.nn.functional.cosine_similarity(x_flat, full_x_flat, dim=1)
    # 返回平均cosine相似度误差 (1 - cosine_similarity)
    return (1 - cosine_sim.mean()).item()

def calculate_all_errors(x, full_x, eps=1e-10):
    """
    计算所有误差指标
    返回字典包含所有误差值
    """
    errors = {}
    errors['l1'] = calculate_l1_error(x, full_x)
    errors['l2'] = calculate_l2_error(x, full_x)
    errors['relative_l1'] = calculate_relative_l1_error(x, full_x, eps)
    errors['relative_l2'] = calculate_relative_l2_error(x, full_x, eps)
    errors['cosine_similarity'] = calculate_cosine_similarity_error(x, full_x, eps)
    
    return errors


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings



class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, current, cache_dic, **kwargs):
        B, N, C = x.shape  


        flops = 0
        test_FLOPs = cache_dic.get('test_FLOPs', False)  
        check_layer = (current['layer'] == 27)
        full_x = x.clone()
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        if test_FLOPs:
            flops += B * C  # SiLU FLOPs
            flops += B * C * 6 * C  # Linear FLOPs in adaLN_modulation

        if current['type'] == 'full':  
            if test_FLOPs:
                flops += 2 * B * N * C
            
            current['module'] = 'attn'
            taylor_cache_init(cache_dic, current)
            attn_output = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), cache_dic=cache_dic, current=current)
            derivative_approximation(cache_dic, current, attn_output)
            x = x + gate_msa.unsqueeze(1) * attn_output


            current['module'] = 'mlp'
            taylor_cache_init(cache_dic, current)
            mlp_output = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            derivative_approximation(cache_dic, current, mlp_output)
            x = x + gate_mlp.unsqueeze(1) * mlp_output

            if test_FLOPs:
                mlp_hidden_dim = int(C * 4)  
                flops += B * N * C * mlp_hidden_dim * 2  
                flops += B * N * mlp_hidden_dim * C * 2 
                flops += B * N * mlp_hidden_dim * 6  

        elif current['type'] == 'Taylor':  
            
            x = cache_step(x=x,
                           sa_dict = cache_dic['cache'][-1][current['layer']]['attn'],
                           mlp_dict= cache_dic['cache'][-1][current['layer']]['mlp'],
                           gate_msa = gate_msa, gate_mlp = gate_mlp,
                           distance = current['step'] - current['activated_steps'][-1])

            if check_layer and cache_dic['check']:
                attn_output = self.attn(modulate(self.norm1(full_x), shift_msa, scale_msa), cache_dic, current) 
                full_x = full_x + gate_msa.unsqueeze(1) * attn_output
                
                mlp_output = self.mlp(modulate(self.norm2(full_x), shift_mlp, scale_mlp))
                full_x = full_x + gate_mlp.unsqueeze(1) * mlp_output
                
                if test_FLOPs:
                    mlp_hidden_dim = int(C * 4)  
                    flops += B * N * C * mlp_hidden_dim * 2  
                    flops += B * N * mlp_hidden_dim * C * 2 
                    flops += B * N * mlp_hidden_dim * 6  
              
                
                if cache_dic['error_metric'] == 'all':
                    all_errors = calculate_all_errors(x, full_x)
                    current['last_layer_error'] = all_errors['relative_l1']
                else:
                    if cache_dic['error_metric'] == 'l1':
                        error_value = calculate_l1_error(x, full_x)
                    elif cache_dic['error_metric'] == 'l2':
                        error_value = calculate_l2_error(x, full_x)
                    elif cache_dic['error_metric'] == 'relative_l1':
                        error_value = calculate_relative_l1_error(x, full_x)
                    elif cache_dic['error_metric'] == 'relative_l2':
                        error_value = calculate_relative_l2_error(x, full_x)
                    elif cache_dic['error_metric'] == 'cosine_similarity':
                        error_value = calculate_cosine_similarity_error(x, full_x)
                    
                    current['last_layer_error'] = error_value
                    
        
        cache_dic['flops'] += flops

        return x


def cache_add(x, sa, mlp, gate_msa, gate_mlp):
    x = x + gate_msa.unsqueeze(1) * sa
    x = x + gate_mlp.unsqueeze(1) * mlp
    return x

@torch.compile
def cache_step(x:torch.Tensor, sa_dict: dict, mlp_dict: dict, gate_msa, gate_mlp, distance: int):
    seer_attn = taylor_formula(sa_dict, distance)
    seer_mlp = taylor_formula(mlp_dict, distance)

    x = cache_add(x, seer_attn, seer_mlp, gate_msa, gate_mlp)

    return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self.layer_outputs = []
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs


    def forward(self, x, t, current, cache_dic, y): 

        self.layer_outputs = []
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        
        cal_type(cache_dic, current)
        for layeridx, block in enumerate(self.blocks):
            current['layer'] = layeridx
            x = block(x, c, current, cache_dic)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x
    
    def forward_with_cfg(self, x, t, current, cache_dic, y, cfg_scale, **kwargs):
    #def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        #model_out = self.forward(combined, t, y)
        model_out = self.forward(combined, t, current, cache_dic, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


#################################################################################
#                           误差计算使用示例                                    #
#################################################################################

def example_error_calculation():
    """
    误差计算函数使用示例
    """
    # 创建示例张量
    x = torch.randn(2, 10, 256)  # 近似结果
    full_x = torch.randn(2, 10, 256)  # 完整结果
    
    # 方法1: 计算单个误差指标
    l1_error = calculate_l1_error(x, full_x)
    l2_error = calculate_l2_error(x, full_x)
    rel_l1_error = calculate_relative_l1_error(x, full_x)
    rel_l2_error = calculate_relative_l2_error(x, full_x)
    cosine_error = calculate_cosine_similarity_error(x, full_x)
    
    print(f"L1误差: {l1_error:.6f}")
    print(f"L2误差: {l2_error:.6f}")
    print(f"相对L1误差: {rel_l1_error:.6f}")
    print(f"相对L2误差: {rel_l2_error:.6f}")
    print(f"Cosine相似度误差: {cosine_error:.6f}")
    
    # 方法2: 一次性计算所有误差指标
    all_errors = calculate_all_errors(x, full_x)
    print("\n所有误差指标:")
    for metric, value in all_errors.items():
        print(f"{metric}: {value:.6f}")
    
    return all_errors

# 使用示例（取消注释以运行）
# if __name__ == "__main__":
#     example_error_calculation()
