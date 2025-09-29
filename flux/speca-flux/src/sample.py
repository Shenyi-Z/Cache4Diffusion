import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from einops import rearrange
from PIL import ExifTags, Image
from transformers import pipeline
from tqdm import tqdm

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, denoise_test_FLOPs
from flux.ideas import denoise_cache
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5

NSFW_THRESHOLD = 0.85  # NSFW score threshold


@dataclass
class SamplingOptions:
    prompts: list[str]          # List of prompts
    width: int                  # Image width
    height: int                 # Image height
    num_steps: int              # Number of sampling steps
    guidance: float             # Guidance value
    seed: int | None            # Random seed
    num_images_per_prompt: int  # Number of images generated per prompt
    batch_size: int             # Batch size (batching of prompts)
    model_name: str             # Model name
    output_dir: str             # Output directory
    add_sampling_metadata: bool # Whether to add metadata
    use_nsfw_filter: bool       # Whether to enable NSFW filter
    test_FLOPs: bool            # Whether in FLOPs test mode (no actual image generation)

    base_threshold: float
    decay_rate: float
    min_taylor_steps: int
    max_taylor_steps: int



def main(opts: SamplingOptions):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Optional NSFW classifier
    if opts.use_nsfw_filter:
        nsfw_classifier = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=device
        )
    else:
        nsfw_classifier = None

    # Load model
    model_name = opts.model_name
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model name: {model_name}, available options: {available}")

    if opts.num_steps is None:
        opts.num_steps = 4 if model_name == "flux-schnell" else 50

    # Ensure width and height are multiples of 16
    opts.width = 16 * (opts.width // 16)
    opts.height = 16 * (opts.height // 16)

    # Set output directory and index
    output_name = os.path.join(opts.output_dir, f"img_{{idx}}.jpg")
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    idx = 0  # Image index

    # Initialize model components
    torch_device = device

    # Load T5 and CLIP models to GPU
    t5 = load_t5(torch_device, max_length=256 if model_name == "flux-schnell" else 512)
    clip = load_clip(torch_device)

    # Load model to GPU
    model = load_flow_model(model_name, device=torch_device)
    ae = load_ae(model_name, device=torch_device)

    # Set random seed
    if opts.seed is not None:
        base_seed = opts.seed
    else:
        base_seed = torch.randint(0, 2**32, (1,)).item()

    prompts = opts.prompts

    total_images = len(prompts) * opts.num_images_per_prompt

    progress_bar = tqdm(total=total_images, desc="Generating images") 

    # Compute number of prompt batches
    num_prompt_batches = (len(prompts) + opts.batch_size - 1) // opts.batch_size

    for batch_idx in range(num_prompt_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(prompts))
        batch_prompts = prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        # Generate corresponding number of images for each prompt
        for image_idx in range(opts.num_images_per_prompt):
            # Prepare random seed
            seed = base_seed + idx  # Assign a different seed for each image
            idx += num_prompts_in_batch  # Update image index

            # Prepare input
            batch_size = num_prompts_in_batch
            x = get_noise(
                batch_size,
                opts.height,
                opts.width,
                device=torch_device,
                dtype=torch.bfloat16,
                seed=seed,
            )

            # Prepare prompts
            # batch_prompts is a list containing the prompts in the current batch
            inp = prepare(t5, clip, x, prompt=batch_prompts)
            timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell"))
            
            # Denoising
            with torch.no_grad():
                if opts.test_FLOPs:
                    x = denoise_test_FLOPs(model, **inp, timesteps=timesteps, guidance=opts.guidance,
                                           base_threshold=opts.base_threshold,
                                            decay_rate=opts.decay_rate,
                                            min_taylor_steps=opts.min_taylor_steps,
                                            max_taylor_steps=opts.max_taylor_steps)
                else:
                    x = denoise_cache(model, **inp, timesteps=timesteps, guidance=opts.guidance, 
                                            base_threshold=opts.base_threshold,
                                            decay_rate=opts.decay_rate,
                                            min_taylor_steps=opts.min_taylor_steps,
                                            max_taylor_steps=opts.max_taylor_steps)
                    # x = search_denoise_cache(model, **inp, timesteps=timesteps, guidance=opts.guidance, interval=opts.interval, max_order=opts.max_order, first_enhance=opts.first_enhance)

                # Decode latent variables
                x = unpack(x.float(), opts.height, opts.width)
                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

            # Convert to PIL format and save
            x = x.clamp(-1, 1)
            x = embed_watermark(x.float())
            x = rearrange(x, "b c h w -> b h w c")

            for i in range(batch_size):
                img_array = x[i]
                img = Image.fromarray((127.5 * (img_array + 1.0)).cpu().byte().numpy())

                # Optional NSFW filtering
                if opts.use_nsfw_filter:
                    nsfw_result = nsfw_classifier(img)
                    nsfw_score = next((res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0)
                else:
                    nsfw_score = 0.0  # If the filter is not enabled, assume safe

                if nsfw_score < NSFW_THRESHOLD:
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                    exif_data[ExifTags.Base.Model] = model_name
                    if opts.add_sampling_metadata:
                        exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]
                    # Save image
                    fn = output_name.format(idx=idx - num_prompts_in_batch + i)
                    img.save(fn, exif=exif_data, quality=95, subsampling=0)
                else:
                    print(f"Generated image may contain inappropriate content, skipped.")

                progress_bar.update(1)

    progress_bar.close()


def read_prompts(prompt_file: str):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def app():
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using the flux model.")
    parser.add_argument('--prompt_file', type=str, default='./test.txt', help='Path to the prompt text file.')
    parser.add_argument('--width', type=int, default=1024, help='Width of the generated image.')
    parser.add_argument('--height', type=int, default=1024, help='Height of the generated image.')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of sampling steps.')
    parser.add_argument('--guidance', type=float, default=3.5, help='Guidance value.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--num_images_per_prompt', type=int, default=1, help='Number of images per prompt.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (prompt batching).')
    parser.add_argument('--model_name', type=str, default='flux-dev', choices=['flux-dev', 'flux-schnell'], help='Model name.')
    parser.add_argument('--output_dir', type=str, default='./samples', help='Directory to save images.')
    parser.add_argument('--add_sampling_metadata', action='store_true', help='Whether to add prompt metadata to images.')
    parser.add_argument('--use_nsfw_filter', action='store_true', help='Enable NSFW filter.')
    parser.add_argument('--test_FLOPs', action='store_true', help='Test inference computation cost.')

    parser.add_argument("--base-threshold",  type=float, default=7)
    parser.add_argument("--decay-rate", type=float, default=0.3) 
    parser.add_argument("--min-taylor-steps", type=int, default=2)
    parser.add_argument("--max-taylor-steps",  type=int, default=8)
    

    args = parser.parse_args()

    prompts = read_prompts(args.prompt_file)

    opts = SamplingOptions(
        prompts=prompts,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        model_name=args.model_name,
        output_dir=args.output_dir,
        add_sampling_metadata=args.add_sampling_metadata,
        use_nsfw_filter=args.use_nsfw_filter,
        test_FLOPs=args.test_FLOPs,

        base_threshold=args.base_threshold,
        decay_rate=args.decay_rate,
        min_taylor_steps=args.min_taylor_steps,
        max_taylor_steps=args.max_taylor_steps,
    )


    main(opts)


if __name__ == '__main__':
    app()
