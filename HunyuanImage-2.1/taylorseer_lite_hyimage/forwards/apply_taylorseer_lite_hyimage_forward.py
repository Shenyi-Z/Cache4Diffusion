import torch
from typing import Optional, Union, Dict
from types import MethodType
from taylorseer_lite_hyimage.cache_utils import cal_type
from taylorseer_lite_hyimage.taylorseer_utils import derivative_approximation, taylor_formula, taylor_cache_init
import loguru
def apply_taylorseer_lite_hyimage_forward(model):
    """
    Apply TaylorSeer Lite HyImage forward.
    """
    from hyimage.models.hunyuan.modules.flash_attn_no_pad import get_cu_seqlens
    
    def taylorseer_lite_hyimage_forward(
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
        cache_dic=None,
        current=None,
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

        cal_type(cache_dic, current)
        current['stream'] = 'final'
        current['layer'] = 'final'
        current['module'] = 'final'
        taylor_cache_init(cache_dic, current)

        if current['type'] == 'full':
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
            derivative_approximation(cache_dic, current, img)
    
        elif current['type'] == 'Taylor':
            img = taylor_formula(cache_dic, current)

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

    model.forward = MethodType(taylorseer_lite_hyimage_forward, model)
    loguru.logger.info("TaylorSeer Lite HyImage forward applied")