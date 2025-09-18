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

### a. TaylorSeer(ICCV 2025)
original repo:  [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)

Taylorseer is an upgraded version of the traditional feature caching method, evolving from the original "cache then reuse" paradigm to the "cache then forecast" paradigm, indicating that the features of diffusion models can be predicted. For example, in this work, they use a simple Taylor series formula to predict features, easily achieving approximately a 5 $\times$ speedup on models like Flux, Qwen-Image and HunyuanVideo.

### b. SpeCa (ACM MM 2025)

SpeCa represents a further advancement beyond TaylorSeer: we recognize that the generation difficulty varies across different samples, making it necessary to adaptively adjust computational costs based on sample complexity. Drawing inspiration from the concept of speculative decoding in language models, we introduce it into diffusion models and employ TaylorSeer as a "draft model" capable of providing high-speed inference, thereby achieving further breakthroughs in acceleration. For instance, on models such as Flux and HunyuanVideo, we have achieved nearly or even exceeding a 6x speedup.

### c. ClusCa (ACM MM 2025)

ClusCa accelerates diffusion models by jointly exploiting spatial and temporal token similarities instead of previous temporal-only feature caching. Guided by spatial clustering, our approach selectively computes a minimal subset of tokens (as few as 16) at intermediate denoising steps. Feature updates from this subset are then efficiently propagated to all tokens, a mechanism that critically mitigates error accumulation caused by prolonged caching. Experimental results demonstrate that ClusCa achieves significant quality improvements while maintaining competitive acceleration performance (e.g. 0.9949 Image Reward at 4.5 $\times$ speedup).


## TODO List:

- [x] Fully support HunyuanVideo
- [ ] Support Wan2.2
- [ ] Support HunyuanImage2.1
