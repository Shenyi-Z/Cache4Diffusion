"""
Reference code
[FLUX] https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py
[DCAE] https://github.com/mit-han-lab/efficientvit/blob/master/efficientvit/models/efficientvit/dc_ae.py
"""

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch import Tensor, nn

from .hunyuanimage_vae import BaseOutput, DiagonalGaussianDistribution


@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.FloatTensor
    posterior: Optional[DiagonalGaussianDistribution] = None


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def forward_with_checkpointing(module, *inputs, use_checkpointing=False):
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    if use_checkpointing:
        return torch.utils.checkpoint.checkpoint(create_custom_forward(module), *inputs, use_reentrant=False)
    else:
        return module(*inputs)


# Optimized implementation of CogVideoXSafeConv3d
# https://github.com/huggingface/diffusers/blob/c9ff360966327ace3faad3807dc871a4e5447501/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py#L38
class PatchCausalConv3d(nn.Conv3d):
    def find_split_indices(self, seq_len, part_num):
        ideal_interval = seq_len / part_num
        possible_indices = list(range(0, seq_len, self.stride[0]))
        selected_indices = []

        for i in range(1, part_num):
            closest = min(possible_indices, key=lambda x: abs(x - round(i * ideal_interval)))
            if closest not in selected_indices:
                selected_indices.append(closest)

        merged_indices = []
        prev_idx = 0
        for idx in selected_indices:
            if idx - prev_idx >= self.kernel_size[0]:
                merged_indices.append(idx)
                prev_idx = idx

        return merged_indices

    def forward(self, input):
        T = input.shape[2]  # input: NCTHW
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024**3
        if T > self.kernel_size[0] and memory_count > 2:
            kernel_size = self.kernel_size[0]
            part_num = int(memory_count / 2) + 1
            split_indices = self.find_split_indices(T, part_num)
            input_chunks = torch.tensor_split(input, split_indices, dim=2) if len(split_indices) > 0 else [input]
            if kernel_size > 1:
                input_chunks = [input_chunks[0]] + [
                    torch.cat(
                        (
                            input_chunks[i - 1][:, :, -kernel_size + 1 :],
                            input_chunks[i],
                        ),
                        dim=2,
                    )
                    for i in range(1, len(input_chunks))
                ]

            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super().forward(input_chunk))
            output = torch.cat(output_chunks, dim=2)
            return output
        else:
            return super().forward(input)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode="replicate",
        disable_causal=False,
        enable_patch_conv=False,
        **kwargs,
    ):
        super().__init__()

        self.pad_mode = pad_mode
        if disable_causal:
            padding = (
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
            )
        else:
            padding = (
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size - 1,
                0,
            )  # W, H, T
        self.time_causal_padding = padding

        if enable_patch_conv:
            self.conv = PatchCausalConv3d(
                chan_in,
                chan_out,
                kernel_size,
                stride=stride,
                dilation=dilation,
                **kwargs,
            )
        else:
            self.conv = nn.Conv3d(
                chan_in,
                chan_out,
                kernel_size,
                stride=stride,
                dilation=dilation,
                **kwargs,
            )

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Conv3d(nn.Conv3d):
    """Perform Conv3d on patches with numerical differences from nn.Conv3d within 1e-5. Only symmetric padding is supported."""

    def forward(self, input):
        B, C, T, H, W = input.shape
        memory_count = (C * T * H * W) * 2 / 1024**3
        if memory_count > 2:
            n_split = math.ceil(memory_count / 2)
            assert n_split >= 2
            chunks = torch.chunk(input, chunks=n_split, dim=-3)
            padded_chunks = []
            for i in range(len(chunks)):
                if self.padding[0] > 0:
                    padded_chunk = F.pad(
                        chunks[i],
                        (0, 0, 0, 0, self.padding[0], self.padding[0]),
                        mode="constant" if self.padding_mode == "zeros" else self.padding_mode,
                        value=0,
                    )
                    if i > 0:
                        padded_chunk[:, :, : self.padding[0]] = chunks[i - 1][:, :, -self.padding[0] :]
                    if i < len(chunks) - 1:
                        padded_chunk[:, :, -self.padding[0] :] = chunks[i + 1][:, :, : self.padding[0]]
                else:
                    padded_chunk = chunks[i]
                padded_chunks.append(padded_chunk)
            padding_bak = self.padding
            self.padding = (0, self.padding[1], self.padding[2])
            outputs = []
            for i in range(len(padded_chunks)):
                outputs.append(super().forward(padded_chunks[i]))
            self.padding = padding_bak
            return torch.cat(outputs, dim=-3)
        else:
            return super().forward(input)


def prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, device, batch_size: int = None):
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.norm = RMS_norm(in_channels, images=False)

        self.q = Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        attention_mask = prepare_causal_attention_mask(f, h * w, h_.dtype, h_.device, batch_size=b)
        h_ = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.unsqueeze(1))

        return rearrange(h_, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        # self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = RMS_norm(in_channels, images=False)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)

        # self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        # self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = RMS_norm(out_channels, images=False)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        self.add_temporal_downsample = add_temporal_downsample
        stride = (2, 2, 2) if add_temporal_downsample else (1, 2, 2)  # THW
        # no asymmetric padding in torch conv, must do it ourselves
        # self.conv = Conv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=0)
        self.conv = CausalConv3d(in_channels, in_channels, kernel_size=3)

    def forward(self, x: Tensor):
        spatial_pad = (0, 1, 0, 1, 0, 0)  # WHT
        x = nn.functional.pad(x, spatial_pad, mode="constant", value=0)

        temporal_pad = (0, 0, 0, 0, 0, 1) if self.add_temporal_downsample else (0, 0, 0, 0, 1, 1)
        x = nn.functional.pad(x, temporal_pad, mode="replicate")

        x = self.conv(x)
        return x


class DownsampleDCAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        # self.conv = Conv3d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)
        self.conv = CausalConv3d(in_channels, out_channels // factor, kernel_size=3)

        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        if self.add_temporal_downsample:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            h_first = torch.cat([h_first, h_first], dim=1)
            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(
                h_next,
                "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w",
                r1=r1,
                r2=2,
                r3=2,
            )
            h = torch.cat([h_first, h_next], dim=2)
            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            B, C, T, H, W = x_first.shape
            x_first = x_first.view(B, h.shape[1], self.group_size // 2, T, H, W).mean(dim=2)

            x_next = x[:, :, 1:, :, :]
            x_next = rearrange(
                x_next,
                "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w",
                r1=r1,
                r2=2,
                r3=2,
            )
            B, C, T, H, W = x_next.shape
            x_next = x_next.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = shortcut.shape
            shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)

        return h + shortcut


class Upsample(nn.Module):
    def __init__(self, in_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        self.add_temporal_upsample = add_temporal_upsample
        self.scale_factor = (2, 2, 2) if add_temporal_upsample else (1, 2, 2)  # THW
        # self.conv = Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv = CausalConv3d(in_channels, in_channels, kernel_size=3)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class UpsampleDCAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        # self.conv = Conv3d(in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1)
        self.conv = CausalConv3d(in_channels, out_channels * factor, kernel_size=3)

        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        if self.add_temporal_upsample:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            h_first = h_first[:, : h_first.shape[1] // 2]

            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(
                h_next,
                "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)",
                r1=r1,
                r2=2,
                r3=2,
            )
            h = torch.cat([h_first, h_next], dim=2)

            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            x_first = x_first.repeat_interleave(repeats=self.repeats // 2, dim=1)

            x_next = x[:, :, 1:, :, :]
            x_next = rearrange(
                x_next,
                "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)",
                r1=r1,
                r2=2,
                r3=2,
            )
            x_next = x_next.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = torch.cat([x_first, x_next], dim=2)

        else:
            h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = rearrange(
                shortcut,
                "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)",
                r1=r1,
                r2=2,
                r3=2,
            )
        return h + shortcut


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # downsampling
        # self.conv_in = Conv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(
                i_level >= np.log2(ffactor_spatial // ffactor_temporal)
            )
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = DownsampleDCAE(block_in, block_out, add_temporal_downsample)
                block_in = block_out
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        # self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # self.conv_out = Conv3d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)
        self.norm_out = RMS_norm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, 2 * z_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def forward(self, x: Tensor) -> Tensor:
        use_checkpointing = bool(self.training and self.gradient_checkpointing)

        # downsampling
        h = self.conv_in(x)
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                h = forward_with_checkpointing(
                    self.down[i_level].block[i_block],
                    h,
                    use_checkpointing=use_checkpointing,
                )
            if hasattr(self.down[i_level], "downsample"):
                h = forward_with_checkpointing(
                    self.down[i_level].downsample,
                    h,
                    use_checkpointing=use_checkpointing,
                )

        # middle
        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

        # end
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h += shortcut
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # z to block_in
        block_in = block_out_channels[0]
        # self.conv_in = Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block

            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < np.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = UpsampleDCAE(block_in, block_out, add_temporal_upsample)
                block_in = block_out
            self.up.append(up)

        # end
        # self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # self.conv_out = Conv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm_out = RMS_norm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def forward(self, z: Tensor) -> Tensor:
        use_checkpointing = bool(self.training and self.gradient_checkpointing)

        # z to block_in
        repeats = self.block_out_channels[0] // (self.z_channels)
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        # middle
        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

        # upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = forward_with_checkpointing(
                    self.up[i_level].block[i_block],
                    h,
                    use_checkpointing=use_checkpointing,
                )
            if hasattr(self.up[i_level], "upsample"):
                h = forward_with_checkpointing(self.up[i_level].upsample, h, use_checkpointing=use_checkpointing)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class AutoencoderKLConv3D(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float = None,
        shift_factor: Optional[float] = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=latent_channels,
            block_out_channels=block_out_channels,
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            downsample_match_channel=downsample_match_channel,
        )
        self.decoder = Decoder(
            z_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=list(reversed(block_out_channels)),
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            upsample_match_channel=upsample_match_channel,
        )

        self.use_slicing = False
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False
        self.use_tiling_during_training = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // ffactor_temporal
        self.tile_overlap_factor = 0.25

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def enable_tiling_during_training(self, use_tiling: bool = True):
        self.use_tiling_during_training = use_tiling

    def disable_tiling_during_training(self):
        self.enable_tiling_during_training(False)

    def enable_temporal_tiling(self, use_tiling: bool = True):
        self.use_temporal_tiling = use_tiling

    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        self.enable_spatial_tiling(use_tiling)

    def disable_tiling(self):
        self.disable_spatial_tiling()

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def spatial_tiled_encode(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))  # 256 * (1 - 0.25) = 192
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)  # 8 * 0.25 = 2
        row_limit = self.tile_latent_min_size - blend_extent  # 8 - 2 = 6

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_size,
                    j : j + self.tile_sample_min_size,
                ]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        moments = torch.cat(result_rows, dim=-2)
        return moments

    def temporal_tiled_encode(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))  # 64 * (1 - 0.25) = 48
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)  # 8 * 0.25 = 2
        t_limit = self.tile_latent_min_tsize - blend_extent  # 8 - 2 = 6

        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i : i + self.tile_sample_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size
            ):
                tile = self.spatial_tiled_encode(tile)
            else:
                tile = self.encoder(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])
        moments = torch.cat(result_row, dim=-3)
        return moments

    def spatial_tiled_decode(self, z: torch.Tensor):
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))  # 8 * (1 - 0.25) = 6
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)  # 256 * 0.25 = 64
        row_limit = self.tile_sample_min_size - blend_extent  # 256 - 64 = 192

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = z[
                    :,
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=-2)
        return dec

    def temporal_tiled_decode(self, z: torch.Tensor):
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))  # 8 * (1 - 0.25) = 6
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)  # 64 * 0.25 = 16
        t_limit = self.tile_sample_min_tsize - blend_extent  # 64 - 16 = 48
        assert 0 < overlap_size < self.tile_latent_min_tsize

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i : i + self.tile_latent_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size
            ):
                decoded = self.spatial_tiled_decode(tile)
            else:
                decoded = self.decoder(tile)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])
        dec = torch.cat(result_row, dim=-3)
        return dec

    def encode(self, x: Tensor, return_dict: bool = True):
        def _encode(x):
            if self.use_temporal_tiling and x.shape[-3] > self.tile_sample_min_tsize:
                return self.temporal_tiled_encode(x)
            if self.use_spatial_tiling and (
                x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
            ):
                return self.spatial_tiled_encode(x)
            return self.encoder(x)

        assert len(x.shape) == 5  # (B, C, T, H, W)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [_encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = _encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: Tensor, return_dict: bool = True, generator=None):
        def _decode(z):
            if self.use_temporal_tiling and z.shape[-3] > self.tile_latent_min_tsize:
                return self.temporal_tiled_decode(z)
            if self.use_spatial_tiling and (
                z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size
            ):
                return self.spatial_tiled_decode(z)
            return self.decoder(z)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [_decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = _decode(z)

        # if z.shape[-3] == 1:
        #     decoded = decoded[:, :, -1:]

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_posterior: bool = True,
        return_dict: bool = True,
    ):
        posterior = self.encode(sample).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z).sample
        return DecoderOutput(sample=dec, posterior=posterior) if return_dict else (dec, posterior)

    def random_reset_tiling(self, x: torch.Tensor):
        if x.shape[-3] == 1:
            self.disable_spatial_tiling()
            self.disable_temporal_tiling()
            return

        min_sample_size = int(1 / self.tile_overlap_factor) * self.ffactor_spatial
        min_sample_tsize = int(1 / self.tile_overlap_factor) * self.ffactor_temporal
        sample_size = random.choice([None, 1 * min_sample_size, 2 * min_sample_size, 3 * min_sample_size])
        if sample_size is None:
            self.disable_spatial_tiling()
        else:
            self.tile_sample_min_size = sample_size
            self.tile_latent_min_size = sample_size // self.ffactor_spatial
            self.enable_spatial_tiling()

        sample_tsize = random.choice([None, 1 * min_sample_tsize, 2 * min_sample_tsize, 3 * min_sample_tsize])
        if sample_tsize is None:
            self.disable_temporal_tiling()
        else:
            self.tile_sample_min_tsize = sample_tsize
            self.tile_latent_min_tsize = sample_tsize // self.ffactor_temporal
            self.enable_temporal_tiling()
