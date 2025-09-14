import math
from dataclasses import dataclass
from typing import Optional
import torch
from einops import rearrange
from torch import Tensor, nn
import numpy as np
from flux.math import attention, rope

from flux.taylor_utils import taylor_formula, derivative_approximation, taylor_cache_init
from flux.model import calculate_l1_error, calculate_l2_error, calculate_relative_l1_error, calculate_relative_l2_error, calculate_cosine_similarity_error, calculate_all_errors

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )




class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        
        cache_dic = kwargs.get('cache_dic', None)
        current = kwargs.get('current', None)        
        
        if cache_dic is None:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)

            # prepare image for attention
            img_modulated = self.img_norm1(img)
            img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
            img_qkv = self.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

            # prepare txt for attention
            txt_modulated = self.txt_norm1(txt)
            txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
            txt_qkv = self.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

            # run actual attention
            q = torch.cat((txt_q, img_q), dim=2)
            k = torch.cat((txt_k, img_k), dim=2)
            v = torch.cat((txt_v, img_v), dim=2)

            attn = attention(q, k, v, pe=pe)
            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

            # calculate the img bloks
            img = img + img_mod1.gate * self.img_attn.proj(img_attn)
            img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

            # calculate the txt bloks
            txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
            txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        
        else:
            current['stream'] = 'double_stream'

            if (current['type'] == 'full'):    
                img_mod1, img_mod2 = self.img_mod(vec)
                txt_mod1, txt_mod2 = self.txt_mod(vec)

                current['module'] = 'attn'
                
                taylor_cache_init(cache_dic=cache_dic, current=current)
                # prepare image for attention
                img_modulated = self.img_norm1(img)
                img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
                img_qkv = self.img_attn.qkv(img_modulated)
                img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
                
                if cache_dic['cache_type'] == 'k-norm':
                    img_k_norm = img_k.norm(dim=-1, p=2).mean(dim=1)
                    cache_dic['k-norm'][-1][current['stream']][current['layer']]['img_mlp'] = img_k_norm
                elif cache_dic['cache_type'] == 'v-norm':
                    img_v_norm = img_v.norm(dim=-1, p=2).mean(dim=1)
                    cache_dic['v-norm'][-1][current['stream']][current['layer']]['img_mlp'] = img_v_norm
                
                img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

                # prepare txt for attention
                txt_modulated = self.txt_norm1(txt)
                txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
                txt_qkv = self.txt_attn.qkv(txt_modulated)
                txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

                if cache_dic['cache_type'] == 'k-norm':
                    txt_k_norm = txt_k.norm(dim=-1, p=2).mean(dim=1)
                    cache_dic['k-norm'][-1][current['stream']][current['layer']]['txt_mlp'] = txt_k_norm
                elif cache_dic['cache_type'] == 'v-norm':
                    txt_v_norm = txt_v.norm(dim=-1, p=2).mean(dim=1)
                    cache_dic['v-norm'][-1][current['stream']][current['layer']]['txt_mlp'] = txt_v_norm
                
                txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

                # run actual attention
                q = torch.cat((txt_q, img_q), dim=2)
                k = torch.cat((txt_k, img_k), dim=2)
                v = torch.cat((txt_v, img_v), dim=2)

                attn = attention(q, k, v, pe=pe, cache_dic=cache_dic, current=current)


                txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
                cache_dic['txt_shape'] = txt.shape[1]
                
                if cache_dic['cache_type'] == 'attention':
                    cache_dic['attn_map'][-1][current['stream']][current['layer']]['txt_mlp'] = cache_dic['attn_map'][-1][current['stream']][current['layer']]['total'][:, : txt.shape[1]]
                    cache_dic['attn_map'][-1][current['stream']][current['layer']]['img_mlp'] = cache_dic['attn_map'][-1][current['stream']][current['layer']]['total'][:, txt.shape[1] :]

                # calculate the img bloks
                current['module'] = 'img_attn'
                taylor_cache_init(cache_dic=cache_dic, current=current)
                img_attn_out = self.img_attn.proj(img_attn)
                derivative_approximation(cache_dic=cache_dic, current=current, feature=img_attn_out)
                img = img + img_mod1.gate * img_attn_out
                
                current['module'] = 'img_mlp'
                taylor_cache_init(cache_dic=cache_dic, current=current)
                img_mlp_out = self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
                derivative_approximation(cache_dic=cache_dic, current=current, feature=img_mlp_out)
                img = img + img_mod2.gate * img_mlp_out
                

                # calculate the txt bloks
                current['module'] = 'txt_attn'
                taylor_cache_init(cache_dic=cache_dic, current=current)
                txt_attn_out = self.txt_attn.proj(txt_attn)
                derivative_approximation(cache_dic=cache_dic, current=current, feature=txt_attn_out)
                txt = txt + txt_mod1.gate * txt_attn_out

                current['module'] = 'txt_mlp'
                taylor_cache_init(cache_dic=cache_dic, current=current)
                txt_mlp_out = self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
                derivative_approximation(cache_dic=cache_dic, current=current, feature=txt_mlp_out)
                txt = txt + txt_mod2.gate * txt_mlp_out


            elif current['type'] == 'taylor_cache':
                img_mod1, img_mod2 = self.img_mod(vec)
                txt_mod1, txt_mod2 = self.txt_mod(vec)
                
                img, txt = double_cache_step(img, txt,
                                             cache_dic['cache'][-1]['double_stream'][current['layer']]['img_attn'],
                                             cache_dic['cache'][-1]['double_stream'][current['layer']]['img_mlp'],
                                             cache_dic['cache'][-1]['double_stream'][current['layer']]['txt_attn'],
                                             cache_dic['cache'][-1]['double_stream'][current['layer']]['txt_mlp'],
                                             img_mod1, img_mod2,
                                             txt_mod1, txt_mod2,
                                             distance=current['step'] - current['activated_steps'][-1])



        return img, txt

@torch.compile
def double_cache_step(img:torch.Tensor, txt:torch.Tensor, 
                      img_attn_dict:dict,
                      img_mlp_dict:dict,
                      txt_attn_dict:dict,
                      txt_mlp_dict:dict, 
                      img_mod1, img_mod2,
                      txt_mod1, txt_mod2, 
                      distance:int):
    
    img_attn_seer = taylor_formula(img_attn_dict, distance)
    img_mlp_seer = taylor_formula(img_mlp_dict, distance)
    txt_attn_seer = taylor_formula(txt_attn_dict, distance)
    txt_mlp_seer = taylor_formula(txt_mlp_dict, distance)

    img, txt = double_cache_add(img_attn_seer, img_mlp_seer,
                                txt_attn_seer, txt_mlp_seer,
                                img, txt,
                                img_mod1, img_mod2,
                                txt_mod1, txt_mod2)
    return img, txt

def double_cache_add(img_attn_seer:torch.Tensor, img_mlp_seer:torch.Tensor,
                     txt_attn_seer:torch.Tensor, txt_mlp_seer:torch.Tensor,
                     img:torch.Tensor, txt:torch.Tensor,
                     img_mod1, img_mod2,
                     txt_mod1, txt_mod2):
    img = img + img_mod1.gate * img_attn_seer + img_mod2.gate * img_mlp_seer
    txt = txt + txt_mod1.gate * txt_attn_seer + txt_mod2.gate * txt_mlp_seer
    return img, txt  


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)
        # mlp_in
        self.mlp_in = nn.Linear(hidden_size, self.mlp_hidden_dim)

    def load_mlp_in_weights(self, linear1_weight: torch.Tensor, linear1_bias: Optional[torch.Tensor] = None):

        hidden_size = self.hidden_size
        mlp_hidden_dim = self.mlp_hidden_dim
        device = self.linear1.weight.device  

        self.mlp_in.weight = torch.nn.Parameter(linear1_weight[hidden_size * 3:, :].to(device))

        if linear1_bias is not None:

            self.mlp_in.bias = torch.nn.Parameter(linear1_bias[hidden_size * 3:].to(device))

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> Tensor:
        cache_dic = kwargs.get('cache_dic', None)
        current = kwargs.get('current', None)

        mod, _ = self.modulation(vec)
        
        if cache_dic is None:
            x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
            qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            q, k = self.norm(q, k, v)

            # compute attention
            attn = attention(q, k, v, pe=pe, cache_dic=cache_dic, current=current)
            # compute activation in mlp stream, cat again and run second linear layer
            output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        
        else:
            current['stream'] = 'single_stream'
            check_layer = (current['layer'] == 37)
            full_x = x.clone()
            if current['type'] == 'full':

                #cache_dic['cache'][-1]['single_stream'][current['layer']]['mod'] = mod

                x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
                current['module'] = 'mlp'
                qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
                current['module'] = 'attn'
                taylor_cache_init(cache_dic=cache_dic, current=current)
                q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

                if cache_dic['cache_type'] == 'k-norm':
                    cache_dic['k-norm'][-1][current['stream']][current['layer']]['total'] = k.norm(dim=-1, p=2).mean(dim=1)
                elif cache_dic['cache_type'] == 'v-norm':
                    cache_dic['v-norm'][-1][current['stream']][current['layer']]['total'] = v.norm(dim=-1, p=2).mean(dim=1)
                
                q, k = self.norm(q, k, v)

                # compute attention
                attn = attention(q, k, v, pe=pe, cache_dic=cache_dic, current=current)

                cache_dic['cache'][-1]['single_stream'][current['layer']]['attn'] = attn
                # compute activation in mlp stream, cat again and run second linear layer
                current['module'] = 'total'
                taylor_cache_init(cache_dic=cache_dic, current=current)
                output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
                derivative_approximation(cache_dic=cache_dic, current=current, feature=output)
                
                x = x + mod.gate * output

            elif current['type'] == 'taylor_cache':
                current['module'] = 'total'
                
                x, output = single_cache_step(x,
                                              total_dict=cache_dic['cache'][-1]['single_stream'][current['layer']]['total'],
                                              mod=mod,
                                              distance=current['step'] - current['activated_steps'][-1])
                
                if check_layer and cache_dic['check']:
                    x_mod = (1 + mod.scale) * self.pre_norm(full_x) + mod.shift
                    qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
                    q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

                    q, k = self.norm(q, k, v)

                    attn = attention(q, k, v, pe=pe, cache_dic=cache_dic, current=current)
                    output_full = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
                    
                    
                    # 根据参数选择误差计算方法
                    error_metric = cache_dic.get('error_metric', 'relative_l1')
                    
                    if error_metric == 'all':
                        all_errors = calculate_all_errors(output, output_full)
                        current['last_layer_error'] = all_errors['relative_l1']
                    else:
                        if error_metric == 'l1':
                            error_value = calculate_l1_error(output, output_full)
                        elif error_metric == 'l2':
                            error_value = calculate_l2_error(output, output_full)
                        elif error_metric == 'relative_l1':
                            error_value = calculate_relative_l1_error(output, output_full)
                        elif error_metric == 'relative_l2':
                            error_value = calculate_relative_l2_error(output, output_full)
                        elif error_metric == 'cosine_similarity':
                            error_value = calculate_cosine_similarity_error(output, output_full)
                   
                        current['last_layer_error'] = error_value
            
            return x


@torch.compile
def single_cache_step(x:torch.Tensor,
                      total_dict:dict,
                      mod:ModulationOut,
                      distance:int):
    
    total_seer = taylor_formula(total_dict, distance)

    x = single_cache_add(total_seer, x, mod)
    return x, total_seer

def single_cache_add(total_seer:torch.Tensor,
                     x:torch.Tensor,
                     mod:ModulationOut):
    x = x + mod.gate * total_seer
    return x

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
