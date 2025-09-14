import torch

def cache_cutfresh(tokens, cache_dic, current):
    step = current['step']
    layer = current['layer']
    stream = current['stream']
    module = current['module']
    fresh_indices = select_one_fresh_index_per_cluster(cache_dic, current)
    fresh_indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])

    fresh_tokens = torch.gather(input = tokens, dim = 1, index = fresh_indices_expand)
    fresh_tokens = fresh_tokens.to(torch.bfloat16)
    return fresh_indices, fresh_tokens

def select_one_fresh_index_per_cluster(cache_dic, current):
    '''
    select exactly one fresh index per cluster randomly
    '''
    # cluster_info = cache_dic['cluster_info']
    cluster_info = cache_dic['cluster_info'][current['stream']][current['module']]
    cluster_indices, cluster_num, K = cluster_info['cluster_indices'], cluster_info['cluster_num'], cluster_info['k']
    B, N = cluster_indices.shape

    device = cluster_indices.device
    rand_weights = torch.rand((B, N), device=device)

    cluster_ids = torch.arange(cluster_num, device=device).view(1, -1, 1) # [1, 32, 1]

    mask = (cluster_indices.unsqueeze(1) == cluster_ids) # [B, K, N]:[1, 32, 4096]

    masked_weights = torch.where(mask, rand_weights.unsqueeze(1), torch.tensor(-float('inf'), device=device)) # [B, K, N]:[1, 32, 4096]

    fresh_indices = masked_weights.argmax(dim=2, keepdim=False) # [B, K]:[1, 32]

    return fresh_indices

def select_fresh_indices_randomly(tokens, topk):
    '''
    select topk indices randomly (for comparison with ToCa)
    '''
    B, N, D = tokens.shape
    device = tokens.device
    fresh_indices = torch.randn((B, N), device=device).argsort(dim=1)[:, :topk]
    return fresh_indices