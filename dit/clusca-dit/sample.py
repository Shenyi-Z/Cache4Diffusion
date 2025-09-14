# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu" 
    #print("device = ", device, flush=True)
    #print(torch.cuda.device_count(), flush=True)

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"/root/autodl-tmp/pretrained_models/DiT/DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    #vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(f"/root/autodl-tmp/pretrained_models/stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    # change ID number 15 to any other ImageNet category ID
    class_labels = [985]


    # Create sampling noise:
    n = len(class_labels)
    # Sample 4 images for category label
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    #print("cfg scale = ", args.cfg_scale, flush=True)
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    model_kwargs['interval']        = args.interval
    model_kwargs['max_order']       = args.max_order
    model_kwargs['test_FLOPs']      = args.test_FLOPs
    
    model_kwargs['cluster_num']       = args.cluster_num
    model_kwargs['cluster_method']    = args.cluster_method
    model_kwargs['propagation_ratio'] = args.propagation_ratio
    model_kwargs['k']                 = args.k

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    if args.ddim_sample:
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
    else:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
    end.record()
    torch.cuda.synchronize()
    print(f"Total Sampling took {start.elapsed_time(end)*0.001} seconds")

    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--ddim-sample", action="store_true", default=False)
    parser.add_argument("--interval", type=int, default=4) 
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--test-FLOPs", action="store_true", default=False)

    # ClusCa parameters
    parser.add_argument("--cluster-num", type=int, default=16)
    parser.add_argument("--cluster-method", type=str, choices=['kmeans', 'random', 'kmeans++'], default='kmeans')
    parser.add_argument("--propagation-ratio", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=1, help="num of selected fresh tokens per cluster")

    args = parser.parse_args()
    main(args)
