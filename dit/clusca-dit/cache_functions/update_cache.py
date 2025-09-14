import torch
def update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    '''
    Update the cache with the fresh tokens. (used in ToCa)
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']
    
    indices = fresh_indices

    cache_dic['cache'][-1][layer][module][0].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)

def propagation_update_cache(fresh_indices, fresh_tokens, cache_dic, current):
    step = current['step']
    layer = current['layer']
    module = current['module']

    cluster_indices, cluster_num, k = \
        cache_dic['cluster_info']['cluster_indices'], cache_dic['cluster_info']['cluster_num'], cache_dic['cluster_info']['k']
    propagation_ratio = cache_dic['propagation_ratio']
    dim = fresh_tokens.shape[-1]
    cache_dic['cache'][-1][layer][module][0].scatter_(dim=1, index=fresh_indices.unsqueeze(-1).expand(-1, -1, dim), src=fresh_tokens)
    old_cache = cache_dic['cache'][-1][layer][module][0]
    B, N, dim = old_cache.shape
    device = old_cache.device

    fresh_cluster_indices = cluster_indices.gather(dim=1, index=fresh_indices)

    sum_per_cluster = torch.zeros((B, cluster_num, dim), device=device)
    sum_per_cluster.scatter_add_(
        dim=1,
        index=fresh_cluster_indices.unsqueeze(-1).expand(-1, -1, dim),
        src=fresh_tokens.float()
    )

    if k == 1:
        mean_per_cluster = sum_per_cluster                                                       # only when k == 1
    elif k > 1: # we found that when k == 1, we can already obtain effective results. So this branch is not used in final version for efficiency.
        count_per_cluster = torch.zeros((B, cluster_num), device=device)
        count_per_cluster.scatter_add_(
            dim=1,
            index=fresh_cluster_indices,
            src=torch.ones_like(fresh_indices, dtype=torch.float, device=device)
        )
        mean_per_cluster = sum_per_cluster / count_per_cluster.unsqueeze(-1).clamp(min=1e-6)     # for k > 1

    new_cache = mean_per_cluster.gather(1, cluster_indices.unsqueeze(-1).expand(-1, -1, dim))

    cache_dic['cache'][-1][layer][module][0] = new_cache * propagation_ratio + old_cache * (1 - propagation_ratio)