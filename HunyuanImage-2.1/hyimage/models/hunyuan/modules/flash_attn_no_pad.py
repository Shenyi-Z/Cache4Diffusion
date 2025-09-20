import torch
from einops import rearrange

try:
    from flash_attn_interface import flash_attn_varlen_func

    print("Using FlashAttention v3.")
except ImportError:
    print("FlashAttention v3 not found, falling back to v2.")
    from flash_attn import flash_attn_varlen_func

from flash_attn import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input


def get_cu_seqlens(text_mask: torch.Tensor, img_len: int):
    """
    Compute cumulative sequence lengths (cu_seqlens) for FlashAttention.

    Args:
        text_mask (torch.Tensor): Boolean mask of shape (batch_size, text_seq_len).
        img_len (int): Length of image sequence.

    Returns:
        cu_seqlens (torch.Tensor): 1D tensor of cumulative sequence lengths for each segment.
        max_len (int): Maximum sequence length (text + image).
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=text_mask.device)
    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens, max_len


def flash_attn_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_s: int,
    causal: bool = False,
    deterministic: bool = False,
):
    """
    FlashAttention v3 wrapper.

    Args:
        q, k, v (torch.Tensor): Query, key, value tensors of shape (batch, seq, nheads, head_dim).
        cu_seqlens (torch.Tensor): Cumulative sequence lengths.
        max_s (int): Maximum sequence length.
        causal (bool): Whether to apply causal masking.
        deterministic (bool): Deterministic computation.

    Returns:
        torch.Tensor: Output tensor of shape (batch, seq, nheads, head_dim).
    """
    batch_size, seqlen = q.shape[:2]
    q = q.reshape(-1, *q.shape[2:])
    k = k.reshape(-1, *k.shape[2:])
    v = v.reshape(-1, *v.shape[2:])
    output = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_s, max_s, causal=causal, deterministic=deterministic
    )
    output = output.view(batch_size, seqlen, *output.shape[-2:])
    return output


def flash_attn_no_pad(
    qkv: torch.Tensor,
    key_padding_mask: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale=None,
    deterministic: bool = False,
):
    """
    FlashAttention for packed QKV input without padding.

    Args:
        qkv (torch.Tensor): Input tensor of shape (batch, seq, 3, nheads, head_dim).
        key_padding_mask (torch.Tensor): Boolean mask of shape (batch, seq).
        causal (bool): Whether to apply causal masking.
        dropout_p (float): Dropout probability.
        softmax_scale (float, optional): Softmax scaling factor.
        deterministic (bool): Deterministic computation.

    Returns:
        torch.Tensor: Output tensor of shape (batch, seq, nheads, head_dim).
    """
    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    x = rearrange(qkv, "b s three h d -> b s (three h d)")

    # Unpad input for FlashAttention, drop `used_seqlens_in_batch` for version compatibility
    x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)[:4]
    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)

    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    if isinstance(output_unpad, tuple):
        output_unpad = output_unpad[0]

    # Pad output back to original shape
    output = pad_input(
        rearrange(output_unpad, "nnz h d -> nnz (h d)"),
        indices,
        batch_size,
        seqlen,
    )
    output = rearrange(output, "b s (h d) -> b s h d", h=nheads)
    return output
