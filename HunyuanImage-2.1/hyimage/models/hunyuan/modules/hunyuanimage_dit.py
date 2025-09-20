import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin

from hyimage.models.hunyuan.modules.posemb_layers import get_nd_rotary_pos_embed
from hyimage.models.hunyuan.modules.flash_attn_no_pad import get_cu_seqlens

from .activation_layers import get_activation_layer
from .embed_layers import PatchEmbed, PatchEmbed2D, TextProjection, TimestepEmbedder
from .mlp_layers import FinalLayer
from .models import MMDoubleStreamBlock, MMSingleStreamBlock
from .token_refiner import SingleTokenRefiner

from hyimage.models.text_encoder.byT5 import ByT5Mapper


def convert_hunyuan_dict_for_tensor_parallel(state_dict):
    """
    Convert a Hunyuan model state dict to be compatible with tensor parallel architectures.

    Args:
        state_dict: Original state dict

    Returns:
        new_dict: Converted state dict
    """
    new_dict = {}
    for k, w in state_dict.items():
        if k.startswith("double_blocks") and "attn_qkv.weight" in k:
            hidden_size = w.shape[1]
            k1 = k.replace("attn_qkv.weight", "attn_q.weight")
            w1 = w[:hidden_size, :]
            new_dict[k1] = w1
            k2 = k.replace("attn_qkv.weight", "attn_k.weight")
            w2 = w[hidden_size : 2 * hidden_size, :]
            new_dict[k2] = w2
            k3 = k.replace("attn_qkv.weight", "attn_v.weight")
            w3 = w[-hidden_size:, :]
            new_dict[k3] = w3
        elif k.startswith("double_blocks") and "attn_qkv.bias" in k:
            hidden_size = w.shape[0] // 3
            k1 = k.replace("attn_qkv.bias", "attn_q.bias")
            w1 = w[:hidden_size]
            new_dict[k1] = w1
            k2 = k.replace("attn_qkv.bias", "attn_k.bias")
            w2 = w[hidden_size : 2 * hidden_size]
            new_dict[k2] = w2
            k3 = k.replace("attn_qkv.bias", "attn_v.bias")
            w3 = w[-hidden_size:]
            new_dict[k3] = w3
        elif k.startswith("single_blocks") and "linear1" in k:
            hidden_size = state_dict[k.replace("linear1", "linear2")].shape[0]
            k1 = k.replace("linear1", "linear1_q")
            w1 = w[:hidden_size]
            new_dict[k1] = w1
            k2 = k.replace("linear1", "linear1_k")
            w2 = w[hidden_size : 2 * hidden_size]
            new_dict[k2] = w2
            k3 = k.replace("linear1", "linear1_v")
            w3 = w[2 * hidden_size : 3 * hidden_size]
            new_dict[k3] = w3
            k4 = k.replace("linear1", "linear1_mlp")
            w4 = w[3 * hidden_size :]
            new_dict[k4] = w4
        elif k.startswith("single_blocks") and "linear2" in k:
            k1 = k.replace("linear2", "linear2.fc")
            new_dict[k1] = w
        else:
            new_dict[k] = w
    return new_dict


def load_hunyuan_dit_state_dict(model, dit_model_name_or_path, strict=True, assign=False):
    """
    Load a state dict for a Hunyuan model, handling both safetensors and torch formats.

    Args:
        model: Model instance to load weights into
        dit_model_name_or_path: Path to the checkpoint file
        strict: Whether to strictly enforce that the keys in state_dict match the model's keys
        assign: If True, assign weights directly without copying

    Returns:
        model: The model with loaded weights
    """
    from safetensors.torch import load_file as safetensors_load_file

    if not os.path.exists(dit_model_name_or_path):
        raise FileNotFoundError(f"Checkpoint file not found: {dit_model_name_or_path}")

    if dit_model_name_or_path.endswith(".safetensors"):
        state_dict = safetensors_load_file(dit_model_name_or_path)
    else:
        state_dict = torch.load(
            dit_model_name_or_path,
            map_location="cpu",
            weights_only=True,
        )
    try:
        state_dict = convert_hunyuan_dict_for_tensor_parallel(state_dict)
    except Exception:
        pass
    model.load_state_dict(state_dict, strict=strict, assign=assign)
    return model


class HYImageDiffusionTransformer(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        text_states_dim: int = 4096,
        rope_theta: int = 256,
        glyph_byT5_v2: bool = False,
        use_meanflow: bool = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        if hidden_size % heads_num != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}")
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        self.glyph_byT5_v2 = glyph_byT5_v2
        if self.glyph_byT5_v2:
            self.byt5_in = ByT5Mapper(
                in_dim=1472,
                out_dim=2048,
                hidden_dim=2048,
                out_dim1=hidden_size,
                use_residual=False
            )

        # Image projection
        if len(self.patch_size) == 3:
            self.img_in = PatchEmbed(self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs)
        elif len(self.patch_size) == 2:
            self.img_in = PatchEmbed2D(self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs)
        else:
            raise ValueError(f"Unsupported patch_size: {self.patch_size}")

        # Text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                text_states_dim,
                hidden_size,
                heads_num,
                depth=2,
                **factory_kwargs,
            )
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        # Time modulation
        self.time_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)

        # MeanFlow support: only create time_r_in when needed
        self.time_r_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
            if use_meanflow
            else None
        )
        self.use_meanflow = use_meanflow

        # Guidance modulation
        self.guidance_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
            if guidance_embed
            else None
        )

        # Double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # Single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

    def enable_deterministic(self):
        """Enable deterministic mode for all transformer blocks."""
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        """Disable deterministic mode for all transformer blocks."""
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes):
        """
        Get rotary position embeddings for the given sizes.

        Args:
            rope_sizes: Sizes for each rotary dimension.

        Returns:
            freqs_cos, freqs_sin: Cosine and sine frequencies for rotary embedding.
        """
        target_ndim = 3
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def reorder_txt_token(self, byt5_txt, txt, byt5_text_mask, text_mask):
        """
        Reorder text tokens for ByT5 integration.

        Args:
            byt5_txt: ByT5 text embeddings.
            txt: Text embeddings.
            byt5_text_mask: Mask for ByT5 tokens.
            text_mask: Mask for text tokens.

        Returns:
            reorder_txt: Reordered text embeddings.
            reorder_mask: Reordered mask.
        """
        reorder_txt = []
        reorder_mask = []

        for i in range(text_mask.shape[0]):
            byt5_text_mask_i = byt5_text_mask[i].bool()
            text_mask_i = text_mask[i].bool()
            byt5_txt_i = byt5_txt[i]
            txt_i = txt[i]
            reorder_txt_i = torch.cat([
                byt5_txt_i[byt5_text_mask_i],
                txt_i[text_mask_i],
                byt5_txt_i[~byt5_text_mask_i],
                txt_i[~text_mask_i]
            ], dim=0)

            reorder_mask_i = torch.cat([
                byt5_text_mask_i[byt5_text_mask_i],
                text_mask_i[text_mask_i],
                byt5_text_mask_i[~byt5_text_mask_i],
                text_mask_i[~text_mask_i]
            ], dim=0)

            reorder_txt.append(reorder_txt_i)
            reorder_mask.append(reorder_mask_i)

        reorder_txt = torch.stack(reorder_txt)
        reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)

        return reorder_txt, reorder_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        output_features: bool = False,
        output_features_stride: int = 8,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        guidance=None,
        extra_kwargs=None,
        *,
        timesteps_r: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for the transformer.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input image tensor.
        timestep : torch.LongTensor
            Timestep tensor.
        text_states : torch.Tensor
            Text embeddings.
        encoder_attention_mask : torch.Tensor
            Attention mask for text.
        output_features : bool, optional
            Whether to output intermediate features.
        output_features_stride : int, optional
            Stride for outputting features.
        freqs_cos, freqs_sin : torch.Tensor, optional
            Precomputed rotary embeddings.
        return_dict : bool, optional
            Not supported.
        guidance : torch.Tensor, optional
            Guidance vector for distillation.
        extra_kwargs : dict, optional
            Extra arguments for ByT5.
        timesteps_r : torch.LongTensor, optional
            Additional timestep for MeanFlow.

        Returns
        -------
        tuple
            (img, features_list, shape)
        """
        if guidance is None:
            guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)
        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        txt = text_states
        input_shape = x.shape

        # Calculate spatial dimensions and get rotary embeddings
        if len(input_shape) == 5:
            _, _, ot, oh, ow = x.shape
            tt, th, tw = (
                ot // self.patch_size[0],
                oh // self.patch_size[1],
                ow // self.patch_size[2],
            )
            if freqs_cos is None or freqs_sin is None:
                freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))
        elif len(input_shape) == 4:
            _, _, oh, ow = x.shape
            th, tw = (
                oh // self.patch_size[0],
                ow // self.patch_size[1],
            )
            if freqs_cos is None or freqs_sin is None:
                assert freqs_cos is None and freqs_sin is None, "freqs_cos and freqs_sin must be both None or both not None"
                freqs_cos, freqs_sin = self.get_rotary_pos_embed((th, tw))
        else:
            raise ValueError(f"Unsupported hidden_states shape: {x.shape}")

        img = self.img_in(img)

        # Prepare modulation vectors
        vec = self.time_in(t)

        # MeanFlow support: merge timestep and timestep_r if available
        if self.use_meanflow:
            assert self.time_r_in is not None, "use_meanflow is True but time_r_in is None"
        if timesteps_r is not None:
            assert self.time_r_in is not None, "timesteps_r is not None but time_r_in is None"
            vec_r = self.time_r_in(timesteps_r)
            vec = (vec + vec_r) / 2

        # Guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(guidance)

        # Embed image and text
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        if self.glyph_byT5_v2:
            byt5_text_states = extra_kwargs["byt5_text_states"]
            byt5_text_mask = extra_kwargs["byt5_text_mask"]
            byt5_txt = self.byt5_in(byt5_text_states)
            txt, text_mask = self.reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask)

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Calculate cu_seqlens and max_s for flash attention
        cu_seqlens, max_s = get_cu_seqlens(text_mask, img_seq_len)

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # Pass through double stream blocks
        for block in self.double_blocks:
            double_block_args = [img, txt, vec, freqs_cis, text_mask, cu_seqlens, max_s]
            img, txt = block(*double_block_args)

        # Merge txt and img to pass through single stream blocks
        x = torch.cat((img, txt), 1)
        features_list = [] if output_features else None

        if len(self.single_blocks) > 0:
            for index, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    (freqs_cos, freqs_sin),
                    text_mask,
                    cu_seqlens,
                    max_s,
                ]
                x = block(*single_block_args)
                if output_features and index % output_features_stride == 0:
                    features_list.append(x[:, :img_seq_len, ...])

        img = x[:, :img_seq_len, ...]

        # Final layer
        img = self.final_layer(img, vec)

        # Unpatchify based on input shape
        if len(input_shape) == 5:
            img = self.unpatchify(img, tt, th, tw)
            shape = (tt, th, tw)
        elif len(input_shape) == 4:
            img = self.unpatchify_2d(img, th, tw)
            shape = (th, tw)
        else:
            raise ValueError(f"Unsupported input_shape: {input_shape}")

        assert not return_dict, "return_dict is not supported."

        if output_features:
            features_list = torch.stack(features_list, dim=0)
        else:
            features_list = None

        return (img, features_list, shape)

    def unpatchify(self, x, t, h, w):
        """
        Unpatchify 3D tensor.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (N, T, patch_size**2 * C)
        t, h, w: int
            Temporal and spatial dimensions

        Returns
        -------
        torch.Tensor
            Unpatchified tensor of shape (N, C, T*pt, H*ph, W*pw)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def unpatchify_2d(self, x, h, w):
        """
        Unpatchify 2D tensor.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (N, T, patch_size**2 * C)
        h, w: int
            Spatial dimensions

        Returns
        -------
        torch.Tensor
            Unpatchified tensor of shape (N, C, H*ph, W*pw)
        """
        c = self.unpatchify_channels
        ph, pw = self.patch_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, c, ph, pw))
        x = torch.einsum('nhwcpq->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * ph, w * pw))
        return imgs
