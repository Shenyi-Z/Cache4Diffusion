import argparse
import json
import os

import torch
import numpy as np
from PIL import Image, ExifTags
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor

# --- FLUX module related imports ---
from flux.sampling import (
    denoise_test_FLOPs,
    get_noise,
    get_schedule,
    prepare,
    unpack,
)
from flux.ideas import denoise_cache
from flux.util import (
    embed_watermark,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)
from transformers import pipeline

# NSFW threshold (adjust as needed)
NSFW_THRESHOLD = 0.85

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using the FLUX model under the Geneval framework")
    # Required: Input JSONL metadata file, each line must contain at least a "prompt" key
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing metadata for each prompt, each line is a JSON object"
    )
    # FLUX model related parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="flux-schnell",
        choices=["flux-dev", "flux-schnell"],
        help="FLUX model name"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Sampling steps (if not specified: flux-schnell defaults to 4, flux-dev defaults to 50)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1360,
        help="Width of generated images (pixels)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Height of generated images (pixels)"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=3.5,
        help="Conditional guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples per batch when generating images"
    )
    # Output related parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory to save generated results"
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="Skip saving overall grid image"
    )
    # Other options
    parser.add_argument(
        "--add_sampling_metadata",
        action="store_true",
        help="Add prompt text to the metadata of generated images"
    )
    parser.add_argument(
        "--use_nsfw_filter",
        action="store_true",
        help="Enable NSFW content filtering (requires downloading related models)"
    )
    parser.add_argument(
        "--test_FLOPs",
        action="store_true",
        help="Only test inference FLOPs (no actual image generation)"
    )
    parser.add_argument("--base-threshold",  type=float, default=7)
    parser.add_argument("--decay-rate", type=float, default=0.3) 
    parser.add_argument("--min-taylor-steps", type=int, default=2)
    parser.add_argument("--max-taylor-steps",  type=int, default=10)
    
    return parser.parse_args()

def main(args):
    # Read metadata file, each line is a JSON object (must contain at least the "prompt" field)
    with open(args.metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line) for line in fp if line.strip()]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If NSFW filtering is enabled, load the classifier (modify model path or name as needed)
    if args.use_nsfw_filter:
        nsfw_classifier = pipeline(
            "image-classification",
            model="/path/to/your/nsfw_model",  # Replace with the actual NSFW model path
            device=0 if torch.cuda.is_available() else -1
        )
    else:
        nsfw_classifier = None

    # If sampling steps are not specified, set default based on model name
    if args.steps is None:
        args.steps = 4 if args.model_name == "flux-schnell" else 50

    # Ensure image width and height are multiples of 16 (required by FLUX)
    args.width = 16 * (args.width // 16)
    args.height = 16 * (args.height // 16)

    # Load FLUX model components to device (T5, CLIP, Flow model, Autoencoder)
    t5 = load_t5(device, max_length=256 if args.model_name == "flux-schnell" else 512)
    clip = load_clip(device)
    model = load_flow_model(args.model_name, device=device)
    ae = load_ae(args.model_name, device=device)

    # Generate results for each prompt:
    for idx, metadata in enumerate(metadatas):
        prompt = metadata.get("prompt", "")
        print(f"Processing prompt {idx + 1}/{len(metadatas)}: '{prompt}'")

        # Define output directory and sample path
        outpath = os.path.join(args.output_dir, f"{idx:05d}")
        sample_path = os.path.join(outpath, "samples")

        # Create output directories
        os.makedirs(outpath, exist_ok=True)
        os.makedirs(sample_path, exist_ok=True)
        
        # Save current prompt metadata to metadata.jsonl
        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp)
        
        # Initialize progress bar
        pbar = tqdm(total=args.n_samples, desc="Sampling")

        # Generate images for the prompt
        local_index = 0
        while local_index < args.n_samples:
            current_bs = min(args.batch_size, args.n_samples - local_index)
            seed = args.seed + local_index
            x = get_noise(current_bs, args.height, args.width, device=device, dtype=torch.bfloat16, seed=seed)
            prompt_list = [prompt] * current_bs
            inp = prepare(t5, clip, x, prompt=prompt_list)
            timesteps = get_schedule(args.steps, inp["img"].shape[1], shift=(args.model_name != "flux-schnell"))

            with torch.no_grad():
                latent = denoise_cache(model, **inp, timesteps=timesteps, guidance=args.guidance, 
                                            base_threshold=args.base_threshold,
                                            decay_rate=args.decay_rate,
                                            min_taylor_steps=args.min_taylor_steps,
                                            max_taylor_steps=args.max_taylor_steps,)
                latent = unpack(latent.float(), args.height, args.width)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    decoded = ae.decode(latent)
            
            decoded = decoded.clamp(-1, 1)
            decoded = embed_watermark(decoded.float())
            images_tensor = rearrange(decoded, "b c h w -> b h w c")
            
            for i in range(current_bs):
                img_array = (127.5 * (images_tensor[i] + 1.0)).cpu().numpy().astype(np.uint8)
                img = Image.fromarray(img_array)
                img.save(os.path.join(sample_path, f"{local_index:05d}.png"))
                local_index += 1
                pbar.update(1)
        pbar.close()

    print("Generation complete.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
