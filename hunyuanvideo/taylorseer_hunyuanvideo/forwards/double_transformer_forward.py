# TODO: fit for hunyuanvideo
import torch

from typing import Any, Dict, Optional, Tuple, Union

from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformerBlock
import torch

from taylorseer_utils import derivative_approximation, taylor_formula, taylor_cache_init

def taylorseer_hunyuan_video_double_block_forward(
    self: HunyuanVideoTransformerBlock,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    attention_mask: Optional[torch.Tensor] = None,
    freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 1. Input normalization
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )

    joint_attention_kwargs = kwargs.pop("joint_attention_kwargs", {}) or {}

    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']

    if current['type'] == 'full':

        current['module'] = 'attn'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        # encoder_hidden_states -> txt
        # hidden_states -> img
        # (encoder_hidden_states, hidden_states) -> total

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        # 3. Modulation and residual connection
        # 4. Feed-forward

        # Process attention outputs for the `hidden_states`.
        current['module'] = 'img_attn'
        taylor_cache_init(cache_dic=cache_dic, current=current)

        derivative_approximation(cache_dic=cache_dic, current=current, feature=attn_output)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        current['module'] = 'img_mlp'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=ff_output)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        current['module'] = 'txt_attn'
        taylor_cache_init(cache_dic=cache_dic, current=current)

        derivative_approximation(cache_dic=cache_dic, current=current, feature=context_attn_output)
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output

        current['module'] = 'txt_mlp'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=context_ff_output)

        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

    elif current['type'] == 'Taylor':

        current['module'] = 'attn'
        # Attention.
        # symbolic placeholder
        
        # 3. Modulation and residual connection
        # 4. Feed-forward

        # Process attention outputs for the `hidden_states`.
        current['module'] = 'img_attn'

        attn_output = taylor_formula(cache_dic=cache_dic, current=current)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
    
        current['module'] = 'img_mlp'

        ff_output = taylor_formula(cache_dic=cache_dic, current=current)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
    
        # Process attention outputs for the `encoder_hidden_states`.
        current['module'] = 'txt_attn'

        context_attn_output = taylor_formula(cache_dic=cache_dic, current=current)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
    
        current['module'] = 'txt_mlp'

        context_ff_output = taylor_formula(cache_dic=cache_dic, current=current)

        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        
    return encoder_hidden_states, hidden_states