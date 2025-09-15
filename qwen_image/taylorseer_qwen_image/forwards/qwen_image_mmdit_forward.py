# QwenImageTransformerBlock(double_stream) forward with taylorseer
# similar to double stream transformer block of flux

import torch
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from taylorseer_utils import derivative_approximation, taylor_formula, taylor_cache_init

def taylorseer_qwen_image_mmdit_forward(
    self: QwenImageTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Get modulation parameters for both streams
    img_mod_params = self.img_mod(temb)  # [B, 6*dim]
    txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

    # Split modulation parameters for norm1 and norm2
    img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
    txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

    # Process image stream - norm1 + modulation
    img_normed = self.img_norm1(hidden_states)
    img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

    # Process text stream - norm1 + modulation
    txt_normed = self.txt_norm1(encoder_hidden_states)
    txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

    # Use QwenAttnProcessor2_0 for joint attention computation
    # This directly implements the DoubleStreamLayerMegatron logic:
    # 1. Computes QKV for both streams
    # 2. Applies QK normalization and RoPE
    # 3. Concatenates and runs joint attention
    # 4. Splits results back to separate streams
    joint_attention_kwargs = joint_attention_kwargs or {}

    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']

    if current['type'] == 'full':
        # EXPLANATION
        # encoder_hidden_states -> txt
        # hidden_states -> img
        # (encoder_hidden_states, hidden_states) -> total
        
        current['module'] = 'attn'
        taylor_cache_init(cache_dic=cache_dic, current=current)

        # attention
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            # **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Process attention outputs for the `hidden_states` (image stream)
        current['module'] = 'img_attn'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=img_attn_output)
        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output

        current['module'] = 'img_mlp'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=img_mlp_output)

        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # txt stream below
        
        # Process attention outputs for the `encoder_hidden_states` (text stream)
        current['module'] = 'txt_attn'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=txt_attn_output)

        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        current['module'] = 'txt_mlp'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=txt_mlp_output)

        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output


    elif current['type'] == 'Taylor':
        current['stream'] = 'double_stream'
        current['module'] = 'attn'
        # Attention.
        # symbolic placeholder
        
        
        # Process attention outputs for the `hidden_states`.
        current['module'] = 'img_attn'
        attn_output = taylor_formula(cache_dic=cache_dic, current=current)
        hidden_states = hidden_states + img_gate1 * attn_output
        
        current['module'] = 'img_mlp'
        # these two steps contain no parameters, so they are relatively cheap
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        
        img_mlp_output = taylor_formula(cache_dic=cache_dic, current=current)
        hidden_states = hidden_states + img_gate2 * img_mlp_output
        
        # txt stream below
        
        current['module'] = 'txt_attn'
        txt_attn_output = taylor_formula(cache_dic=cache_dic, current=current)
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
        
        current['module'] = 'txt_mlp'
        # these two steps contain no parameters, so they are relatively cheap
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)

        txt_mlp_output = taylor_formula(cache_dic=cache_dic, current=current)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output
        
    else:
        raise ValueError(f"Unknown current['type']: {current['type']} not in options ['full', 'Taylor']")

    # Clip to prevent overflow for fp16
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
        
    return encoder_hidden_states, hidden_states