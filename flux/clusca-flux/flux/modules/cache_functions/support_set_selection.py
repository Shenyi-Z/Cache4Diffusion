import torch
from typing import Dict

def support_set_selection(x: torch.Tensor, fresh_ratio: float, base_ratio: float, current: Dict, cache_dic: Dict) -> torch.Tensor:
    
    B, N, H = x.shape
    num_total = int(fresh_ratio * N) 
    base_count = int(base_ratio * num_total) 
    add_count = num_total - base_count 

    random_indices = torch.randperm(N, device=x.device)
    base_indices = random_indices[:base_count]
    other_indices = random_indices[base_count:]

    base_tokens = x.gather(dim=1, index=base_indices.unsqueeze(-1).expand(B, -1, H))
    
    # normaize
    base_tokens = base_tokens / base_tokens.norm(dim=-1, keepdim=True)
    x_norm = x / x.norm(dim=-1, keepdim=True)

    similarity = torch.einsum('bnd,bmd->bnm', base_tokens, x_norm)

    min_similarity = similarity.min(dim=1).values

    _, min_indices = min_similarity.topk(add_count, largest=False)

    indices = torch.cat([base_indices.expand(B, -1), min_indices], dim=-1) #+ selection_start

    return indices


    