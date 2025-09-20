from hyimage.common.config import LazyCall as L
from hyimage.models.hunyuan.modules.hunyuanimage_dit import HYImageDiffusionTransformer




hunyuanimage_refiner_cfg = L(HYImageDiffusionTransformer)(
    in_channels=128,
    out_channels=64,
    mm_double_blocks_depth=20,
    mm_single_blocks_depth=40,
    rope_dim_list=[16, 56, 56],
    hidden_size=3328,
    heads_num=26,
    mlp_width_ratio=4,
    patch_size=[1, 1, 1],
    text_states_dim=3584,
    guidance_embed=True,
    use_meanflow=True,
)

hunyuanimage_v2_1_cfg = L(HYImageDiffusionTransformer)(
    in_channels=64,
    out_channels=64,
    mm_double_blocks_depth=20,
    mm_single_blocks_depth=40,
    rope_dim_list=[64, 64],
    hidden_size=3584,
    heads_num=28,
    mlp_width_ratio=4,
    patch_size=[1, 1],
    text_states_dim=3584,
    glyph_byT5_v2=True,
    guidance_embed=False,
)

hunyuanimage_v2_1_distilled_cfg = L(HYImageDiffusionTransformer)(
    in_channels=64,
    out_channels=64,
    mm_double_blocks_depth=20,
    mm_single_blocks_depth=40,
    rope_dim_list=[64, 64],
    hidden_size=3584,
    heads_num=28,
    mlp_width_ratio=4,
    patch_size=[1, 1],
    text_states_dim=3584,
    glyph_byT5_v2=True,
    guidance_embed=True,
    use_meanflow=True,
)
