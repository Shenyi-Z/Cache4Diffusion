from dataclasses import dataclass

import torch
from torch import Tensor, nn
import os
from flux.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from flux.modules.lora import LinearLora, replace_linear_with_lora
from flux.modules.cache_functions import cal_type

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

@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool

TOTAL_FULL_COUNTER = 0
TOTAL_CHECK_COUNTER = 0
TOTAL_FLOPS = 0
NUM_IMAGES_GENERATED = 0


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.layer_outputs=[]
    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        *args,
        **kwargs,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        
        cache_dic = kwargs.get('cache_dic', None)
        current = kwargs.get('current', None)
        self.layer_outputs=[]
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        cal_type(cache_dic=cache_dic, current=current)

        for i, block in enumerate(self.double_blocks):
            current['layer'] = i
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, cache_dic=cache_dic, current=current)
            
        img = torch.cat((txt, img), 1)


        for i, block in enumerate(self.single_blocks):
            current['layer'] = i
            img = block(img, vec=vec, pe=pe, cache_dic=cache_dic, current=current)


        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        
        return img

class FluxLoraWrapper(Flux):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)
