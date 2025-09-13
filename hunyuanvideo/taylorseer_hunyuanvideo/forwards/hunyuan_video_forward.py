import logging
import time
import torch
import torch.distributed as dist
from transformers import T5EncoderModel

from typing import Any, Dict, Optional, Tuple, Union
from diffusers import DiffusionPipeline
from diffusers.models import HunyuanVideoTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
import torch
import numpy as np

from cache_functions import cache_init, cal_type

logger = logging.get_logger(__name__)

def taylorseer_hunyuan_video_forward(
    self: HunyuanVideoTransformer3DModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_attention_mask: torch.Tensor = None,  # newly added from hunyuanvideo
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    # img_ids: torch.Tensor = None,
    # txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    # joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,  # diff name in hunyuanvideo
    # controlnet_block_samples=None,
    # controlnet_single_block_samples=None,
    return_dict: bool = True,
    # controlnet_blocks_repeat: bool = False,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    The [`FluxTransformer2DModel`] forward method.
    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.
    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    
    if attention_kwargs is None:
        attention_kwargs = {}
    if attention_kwargs.get("cache_dic", None) is None:
        attention_kwargs['cache_dic'], attention_kwargs['current'] = cache_init(self)

    cal_type(attention_kwargs['cache_dic'], attention_kwargs['current'])

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    # TODO: check if self.config exists in hunyuanvideo. config.json only have "Name" attribute
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = 1 * post_patch_height * post_patch_width

    # 1. RoPE
    image_rotary_emb = self.rope(hidden_states)

    # 2. Conditional embeddings
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    # 3. Attention mask preparation
    latent_sequence_length = hidden_states.shape[1]
    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = torch.ones(
        batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
    )  # [B, N]
    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
    indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)  # [1, N]
    mask_indices = indices >= effective_sequence_length.unsqueeze(1)  # [B, N]
    attention_mask = attention_mask.masked_fill(mask_indices, False)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]

    # timestep = timestep.to(hidden_states.dtype) * 1000
    # if guidance is not None:
    #     guidance = guidance.to(hidden_states.dtype) * 1000
    # else:
    #     guidance = None

    # temb = (
    #     self.time_text_embed(timestep, pooled_projections)
    #     if guidance is None
    #     else self.time_text_embed(timestep, guidance, pooled_projections)
    # )
    # encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # if txt_ids.ndim == 3:
    #     logger.warning(
    #         "Passing `txt_ids` 3d torch.Tensor is deprecated."
    #         "Please remove the batch dimension and pass it as a 2d torch Tensor"
    #     )
    #     txt_ids = txt_ids[0]
    # if img_ids.ndim == 3:
    #     logger.warning(
    #         "Passing `img_ids` 3d torch.Tensor is deprecated."
    #         "Please remove the batch dimension and pass it as a 2d torch Tensor"
    #     )
    #     img_ids = img_ids[0]

    # ids = torch.cat((txt_ids, img_ids), dim=0)
    # image_rotary_emb = self.pos_embed(ids)

    # From impl of hunyuanvideo single transformer block, it has no ip_hidden_states
    # if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
    #     ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
    #     ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
    #     joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})


    # 4. Transformer blocks
    attention_kwargs['current']['stream'] = 'double_stream'

    for index_block, block in enumerate(self.transformer_blocks):

        attention_kwargs['current']['layer'] = index_block

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
                attention_kwargs=attention_kwargs,
            )

        else:                
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
                attention_kwargs=attention_kwargs,
            )
            
        # controlnet residual
        # if controlnet_block_samples is not None:
        #     interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
        #     interval_control = int(np.ceil(interval_control))
        #     # For Xlabs ControlNet.
        #     if controlnet_blocks_repeat:
        #         hidden_states = (
        #             hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
        #         )
        #     else:
        #         hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
    
    attention_kwargs['current']['stream'] = 'single_stream'

    for index_block, block in enumerate(self.single_transformer_blocks):

        attention_kwargs['current']['layer'] = index_block

        if torch.is_grad_enabled() and self.gradient_checkpointing:                
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
                attention_kwargs=attention_kwargs,
            )

        else:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
                attention_kwargs=attention_kwargs,
            )

        # controlnet residual
        # if controlnet_single_block_samples is not None:
        #     interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
        #     interval_control = int(np.ceil(interval_control))
        #     hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
        #         hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        #         + controlnet_single_block_samples[index_block // interval_control]
        #     )

    # 5. Output projection
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)
    
    attention_kwargs['current']['step'] += 1

    if not return_dict:
        return (hidden_states,)

    return Transformer2DModelOutput(sample=hidden_states)