import math
import torch
from typing import Dict, List, Optional, Tuple, Union

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://arxiv.org/pdf/2305.08891.pdf).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.
    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

class ClassifierFreeGuidance:
    def __init__(
        self,
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
    ):
        super().__init__()

        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

    def __call__(self, pred_cond: torch.Tensor, pred_uncond: Optional[torch.Tensor] = None) -> torch.Tensor:

        shift = pred_cond - pred_uncond
        pred = pred_cond if self.use_original_formulation else pred_uncond
        pred = pred + self.guidance_scale * shift

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return pred


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average

def normalized_guidance_apg(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: Optional[MomentumBuffer] = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
    use_original_formulation: bool = False,
):
    diff = pred_cond - pred_uncond
    dim = [-i for i in range(1, len(diff.shape))]

    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dim, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    v0, v1 = diff.double(), pred_cond.double()
    v1 = torch.nn.functional.normalize(v1, dim=dim)
    v0_parallel = (v0 * v1).sum(dim=dim, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    diff_parallel, diff_orthogonal = v0_parallel.type_as(diff), v0_orthogonal.type_as(diff)

    normalized_update = diff_orthogonal + eta * diff_parallel
    pred = pred_cond if use_original_formulation else pred_uncond
    pred = pred + guidance_scale * normalized_update

    return pred

class AdaptiveProjectedGuidance:
    def __init__(
        self,
        guidance_scale: float = 7.5,
        adaptive_projected_guidance_momentum: Optional[float] = None,
        adaptive_projected_guidance_rescale: float = 15.0,
        # eta: float = 1.0,
        eta: float = 0.0,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
    ):
        super().__init__()

        self.guidance_scale = guidance_scale
        self.adaptive_projected_guidance_momentum = adaptive_projected_guidance_momentum
        self.adaptive_projected_guidance_rescale = adaptive_projected_guidance_rescale
        self.eta = eta
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation
        self.momentum_buffer = None

    def __call__(self, pred_cond: torch.Tensor, pred_uncond: Optional[torch.Tensor] = None, step=None) -> torch.Tensor:

        if step == 0 and self.adaptive_projected_guidance_momentum is not None:
            self.momentum_buffer = MomentumBuffer(self.adaptive_projected_guidance_momentum)

        pred = normalized_guidance_apg(
            pred_cond,
            pred_uncond,
            self.guidance_scale,
            self.momentum_buffer,
            self.eta,
            self.adaptive_projected_guidance_rescale,
            self.use_original_formulation,
        )

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return pred
