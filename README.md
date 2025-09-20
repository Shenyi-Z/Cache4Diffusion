# Cache4Diffusion
Aiming to integrate most existing feature caching-based diffusion acceleration schemes into a unified framework.

Over $6 \times$ training-free acceleration on FLUX-dev.
![FLUX](./assets/examples/FLUX.png)
Over $4 \times$ training-free acceleration on Qwen-Image.
![Qwen-Image](./assets/examples/Qwen-Image.png)

## Key Features: 
- More advanced feature cahing-based methods.
- Support both **Nvidia** GPUs and **Ascend** NPUs (see in the folder `./ascend/`).
- Include Text2Image, Text2Video, Class2Image... More models are coming...
- Memory usage analysis supported.

## Currently Supported Methods:
Currently, the project support some of the state-of-the-art (SOTA) acceleration methods:

### a. TaylorSeer(ICCV 2025) <a href='https://arxiv.org/abs/2503.06923'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>

original repo:  [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)

Taylorseer is an upgraded version of the traditional feature caching method, evolving from the original "cache then reuse" paradigm to the "cache then forecast" paradigm, indicating that the features of diffusion models can be predicted. For example, in this work, they use a simple Taylor series formula to predict features, easily achieving approximately a 5 $\times$ speedup on models like Flux, Qwen-Image and HunyuanVideo.

#### TaylorSeer-Lite

TaylorSeer-Lite is a VRAM-optimized version based on TaylorSeer, motivated by the following consideration: TaylorSeer caches the output of each NN in the layer x = x + alpha NN(x). For an order 2 model with 60 blocks, each containing 2 modules (e.g., attn and mlp/moe), this results in caching n_order  n_blocks * n_module = 240 tensors, significantly increasing VRAM usage and reducing deployability. To address this, we modified it to cache only at the output of the final block, thereby eliminating the multiplicative increase in cache size due to n_blocks and n_module. Empirical results show that this modification has little impact on quality while providing substantial VRAM savings and I/O speed improvements—using the Lite version eliminates the need for compilation optimizations to accelerate cache operations, making Feature Caching more practical in real-world applications.

In the support for HunyuanImage-2.1, we adopted TaylorSeer-Lite, resulting in a 5x acceleration in 2k generation. Refer to `HunyuanImage-2.1/taylorseer_hunyuan_image`.

### b. SpeCa (ACM MM 2025) <a href='https://www.arxiv.org/abs/2509.11628'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>

SpeCa represents a further advancement beyond TaylorSeer: we recognize that the generation difficulty varies across different samples, making it necessary to adaptively adjust computational costs based on sample complexity. Drawing inspiration from the concept of speculative decoding in language models, we introduce it into diffusion models and employ TaylorSeer as a "draft model" capable of providing high-speed inference, thereby achieving further breakthroughs in acceleration. For instance, on models such as Flux and HunyuanVideo, we have achieved nearly or even exceeding a 6x speedup.

### c. ClusCa (ACM MM 2025) <a href='https://arxiv.org/abs/2509.10312'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>

ClusCa accelerates diffusion models by jointly exploiting spatial and temporal token similarities instead of previous temporal-only feature caching. Guided by spatial clustering, our approach selectively computes a minimal subset of tokens (as few as 16) at intermediate denoising steps. Feature updates from this subset are then efficiently propagated to all tokens, a mechanism that critically mitigates error accumulation caused by prolonged caching. Experimental results demonstrate that ClusCa achieves significant quality improvements while maintaining competitive acceleration performance (e.g. 0.9949 Image Reward at 4.5 $\times$ speedup).


## TODO List:

- [x] Fully support HunyuanVideo
- [ ] Support Wan2.2
- [x] Support HunyuanImage2.1

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Shenyi-Z/Cache4Diffusion&type=Date)](https://www.star-history.com/#Shenyi-Z/Cache4Diffusion&Date)