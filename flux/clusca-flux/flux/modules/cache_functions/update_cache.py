import torch
def update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    '''
    Update the cache with the fresh tokens.
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']
    # Update the cached tokens at the positions


    indices = fresh_indices

    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][0].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)
    
def propagation_update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    step = current['step']
    layer = current['layer']
    module = current['module']

    fresh_tokens = fresh_tokens.to(torch.bfloat16)
    # cluster_info = cache_dic['cluster_info']
    cluster_info = cache_dic['cluster_info'][current['stream']][current['module']]
    cluster_indices, cluster_num, k = \
        cluster_info['cluster_indices'], cluster_info['cluster_num'], cluster_info['k']
    propagation_ratio = cache_dic['propagation_ratio']
    dim = fresh_tokens.shape[-1]
    # cache_dic['cache'][-1][current['stream']][layer][module][0].scatter_(dim=1, index=fresh_indices.unsqueeze(-1).expand(-1, -1, dim), src=fresh_tokens)
    # old_cache = cache_dic['cache'][-1][current['stream']][layer][module][0]
    # B, N, dim = old_cache.shape
    # device = old_cache.device
    # fresh_cluster_indices = cluster_indices.gather(dim=1, index=fresh_indices)
    # sum_per_cluster = torch.zeros((B, cluster_num, dim), device=device, dtype=torch.bfloat16)
    # sum_per_cluster.scatter_add_(
    #     dim=1,
    #     index=fresh_cluster_indices.unsqueeze(-1).expand(-1, -1, dim),
    #     src=fresh_tokens
    # )
    # mean_per_cluster = sum_per_cluster                                                       # only when topk == 1
    # # mean_per_cluster = sum_per_cluster / count_per_cluster.unsqueeze(-1).clamp(min=1e-6)   # when topk > 1

    # new_cache = mean_per_cluster.gather(1, cluster_indices.unsqueeze(-1).expand(-1, -1, dim))
    # cache_dic['cache'][-1][current['stream']][layer][module][0] = new_cache * propagation_ratio + old_cache * (1 - propagation_ratio)
    propagation_update_cache_compile(old_cache_dict=cache_dic['cache'][-1][current['stream']][layer][module],
                                fresh_indices=fresh_indices,
                                fresh_tokens=fresh_tokens,
                                cluster_indices=cluster_indices,
                                cluster_num=cluster_num,
                                k=k,
                                propagation_ratio=propagation_ratio)

@torch.compile
def propagation_update_cache_compile(old_cache_dict, fresh_indices, fresh_tokens, cluster_indices, cluster_num, k, propagation_ratio):
    B, N, dim = old_cache_dict[0].shape
    device = old_cache_dict[0].device
    old_cache_dict[0].scatter_(dim=1, index=fresh_indices.unsqueeze(-1).expand(-1, -1, dim), src=fresh_tokens)
    fresh_cluster_indices = cluster_indices.gather(dim=1, index=fresh_indices)
    sum_per_cluster = torch.zeros((B, cluster_num, dim), device=device, dtype=torch.bfloat16)
    sum_per_cluster.scatter_add_(
        dim=1,
        index=fresh_cluster_indices.unsqueeze(-1).expand(-1, -1, dim),
        src=fresh_tokens
        )
    if k == 1:
        mean_per_cluster = sum_per_cluster                                                       # only when topk == 1
    elif k > 1: # we found that when k == 1, we can already obtain effective results. So this branch is not used in final version for efficiency.
        count_per_cluster = torch.zeros((B, cluster_num), device=device)
        count_per_cluster.scatter_add_(
            dim=1,
            index=fresh_cluster_indices,
            src=torch.ones_like(fresh_indices, dtype=torch.float, device=device)
        )
        mean_per_cluster = sum_per_cluster / count_per_cluster.unsqueeze(-1).clamp(min=1e-6)     # for k > 1

    new_cache = mean_per_cluster.gather(1, cluster_indices.unsqueeze(-1).expand(-1, -1, dim))
    old_cache_dict[0] = new_cache * propagation_ratio + old_cache_dict[0] * (1 - propagation_ratio)