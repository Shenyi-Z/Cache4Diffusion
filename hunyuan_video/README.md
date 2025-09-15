For the environment, basically, you can follow the official settings.

But we recommend to install `diffusers=0.36.0.dev0` (currently latest).

And the modeled used to generate video is `hunyuanvideo-community/HunyuanVideo` for better `diffusers` support.

Using `dtype=bfloat16` is highly recommended. `float16` may lead to `nan` output and black video according to our testing.