import torch
from typing import Optional
from PIL import Image
from tqdm import tqdm
from taylorseer_lite_hyimage.cache_utils import cache_init
from types import MethodType
import loguru
def apply_taylorseer_lite_hyimage_pipeline(pipeline):
    """
    Apply taylorseer lite to HunyuanImage pipeline.
    """
    def _denoise_step_taylorseer_lite(self, latents, timesteps, text_emb, text_mask, byt5_emb, byt5_mask, guidance_scale: float = 1.0, timesteps_r=None, cache_dic=None, current=None):
        """
        Perform one denoising step.

        Args:
            latents: Latent tensor
            timesteps: Timesteps tensor
            text_emb: Text embedding
            text_mask: Text mask
            byt5_emb: byT5 embedding
            byt5_mask: byT5 mask
            guidance_scale: Guidance scale
            timesteps_r: Optional next timestep

        Returns:
            Noise prediction tensor
        """
        if byt5_emb is not None and byt5_mask is not None:
            extra_kwargs = {
                "byt5_text_states": byt5_emb,
                "byt5_text_mask": byt5_mask,
            }
        else:
            if self.use_byt5:
                raise ValueError("Must provide byt5_emb and byt5_mask for HunyuanImage 2.1")
            extra_kwargs = {}

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            if hasattr(self.dit, 'guidance_embed') and self.dit.guidance_embed:
                guidance_expand = torch.tensor(
                    [guidance_scale] * latents.shape[0],
                    dtype=torch.float32,
                    device=latents.device
                ).to(latents.dtype) * 1000
            else:
                guidance_expand = None

            noise_pred = self.dit(
                latents,
                timesteps,
                text_states=text_emb,
                encoder_attention_mask=text_mask,
                guidance=guidance_expand,
                return_dict=False,
                extra_kwargs=extra_kwargs,
                timesteps_r=timesteps_r,
                cache_dic=cache_dic,
                current=current,
            )[0]

        return noise_pred

    @torch.no_grad()
    def __call_taylorseer_lite_pipeline(
        self,
        prompt: str,
        shift: int = 5,
        negative_prompt: str = "",
        width: int = 2048,
        height: int = 2048,
        use_reprompt: bool = False,
        use_refiner: bool = False,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = 42,
        **kwargs
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text prompt describing the image
            negative_prompt: Negative prompt for guidance
            width: Image width
            height: Image height
            use_reprompt: Whether to use reprompt model
            use_refiner: Whether to use refiner pipeline
            num_inference_steps: Number of denoising steps (overrides config if provided)
            guidance_scale: Strength of classifier-free guidance (overrides config if provided)
            seed: Random seed for reproducibility
            **kwargs: Additional arguments

        Returns:
            Generated PIL Image
        """
        if seed is not None:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            torch.manual_seed(seed)
        else:
            generator = None

        sampling_steps = num_inference_steps if num_inference_steps is not None else self.default_sampling_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.default_guidance_scale
        shift = shift if shift is not None else self.shift

        user_prompt = prompt
        if use_reprompt:
            if self.config.enable_stage1_offloading:
                self.offload()
            prompt = self.reprompt_model.predict(prompt)


        print("=" * 60)
        print("🖼️  HunyuanImage Generation Task")
        print("-" * 60)
        print(f"Prompt:           {user_prompt}")
        if use_reprompt:
            print(f"Reprompt:         {prompt}")
        if not self.cfg_distilled:
            print(f"Negative Prompt:  {negative_prompt if negative_prompt else '(none)'}")
        print(f"Guidance Scale:   {guidance_scale}")
        print(f"CFG Mode:         {self.cfg_mode}")
        print(f"Guidance Rescale: {self.guidance_rescale}")
        print(f"Shift:            {shift}")
        print(f"Seed:             {seed}")
        print(f"Use MeanFlow:     {self.use_meanflow}")
        print(f"Use byT5:         {self.use_byt5}")
        print(f"Image Size:       {width} x {height}")
        print(f"Sampling Steps:   {sampling_steps}")
        print("=" * 60)

        pos_text_emb, pos_text_mask = self._encode_text(prompt)
        neg_text_emb, neg_text_mask = self._encode_text(negative_prompt)

        if self.config.enable_text_encoder_offloading:
            self.text_encoder.to('cpu')

        self.byt5_kwargs['byt5_model'].to(self.execution_device)
        pos_byt5_emb, pos_byt5_mask = self._encode_glyph(prompt)
        neg_byt5_emb, neg_byt5_mask = self._encode_glyph(negative_prompt)
        if self.config.enable_byt5_offloading:
            self.byt5_kwargs['byt5_model'].to('cpu')

        latents = self._prepare_latents(width, height, generator=generator)

        do_classifier_free_guidance = (not self.cfg_distilled) and guidance_scale > 1
        if do_classifier_free_guidance:
            text_emb = torch.cat([neg_text_emb, pos_text_emb])
            text_mask = torch.cat([neg_text_mask, pos_text_mask])

            if self.use_byt5 and pos_byt5_emb is not None and neg_byt5_emb is not None:
                byt5_emb = torch.cat([neg_byt5_emb, pos_byt5_emb])
                byt5_mask = torch.cat([neg_byt5_mask, pos_byt5_mask])
            else:
                byt5_emb = pos_byt5_emb
                byt5_mask = pos_byt5_mask
        else:
            text_emb = pos_text_emb
            text_mask = pos_text_mask
            byt5_emb = pos_byt5_emb
            byt5_mask = pos_byt5_mask

        timesteps, sigmas = self.get_timesteps_sigmas(sampling_steps, shift)

        self.dit.to(self.execution_device)

        cache_dic, current = cache_init(sampling_steps)

        for i, t in enumerate(tqdm(timesteps, desc="Denoising", total=len(timesteps))):
            current['step'] = i
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            t_expand = t.repeat(latent_model_input.shape[0])
            if self.use_meanflow:
                if i == len(timesteps) - 1:
                    timesteps_r = torch.tensor([0.0], device=self.execution_device)
                else:
                    timesteps_r = timesteps[i + 1]
                timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
            else:
                timesteps_r = None

            if self.cfg_distilled:
                noise_pred = self._denoise_step_taylorseer_lite(
                    latent_model_input, t_expand, text_emb, text_mask, byt5_emb, byt5_mask, guidance_scale, timesteps_r=timesteps_r, cache_dic=cache_dic, current=current,
                )
            else:
                noise_pred = self._denoise_step_taylorseer_lite(
                    latent_model_input, t_expand, text_emb, text_mask, byt5_emb, byt5_mask, timesteps_r=timesteps_r, cache_dic=cache_dic, current=current,
                )

            if do_classifier_free_guidance:
                noise_pred = self._apply_classifier_free_guidance(noise_pred, guidance_scale, i)

            latents = self.step(latents, noise_pred, sigmas, i)


        if self.config.enable_full_dit_offloading:
            self.dit.to('cpu')
        self.vae.to(self.execution_device)
        image = self._decode_latents(latents)
        if self.config.enable_vae_offloading:
            self.vae.to('cpu')
        image = (image.squeeze(0).permute(1, 2, 0) * 255).byte().numpy()
        pil_image = Image.fromarray(image)
        stats = torch.cuda.memory_stats()
        peak_bytes_requirement = stats["allocated_bytes.all.peak"]
        print(f"Before refiner Peak memory requirement: {peak_bytes_requirement / 1024 ** 3:.2f} GB")

        if use_refiner:
            if self.config.enable_stage1_offloading:
                self.offload()
            pil_image = self.refiner_pipeline(
                image=pil_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                use_reprompt=False,
                use_refiner=False,
                num_inference_steps=4,
                guidance_scale=guidance_scale,
                generator=generator,
                seed=seed,
            )
            if self.config.enable_refiner_offloading:
                self.refiner_pipeline.offload()

        return pil_image

    pipeline._denoise_step_taylorseer_lite = MethodType(_denoise_step_taylorseer_lite, pipeline)
    
    # Replace the class __call__ method to make pipeline() calls work correctly
    # This is necessary because Python looks up __call__ in the class, not the instance
    pipeline.__class__.__call__ = __call_taylorseer_lite_pipeline
    
    loguru.logger.info("TaylorSeer Lite HyImage pipeline applied")