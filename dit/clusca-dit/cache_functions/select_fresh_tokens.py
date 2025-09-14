import torch

def select_one_fresh_index_per_cluster(cache_dic, current):
    '''
    select exactly one fresh index per cluster randomly
    '''
    cluster_info = cache_dic['cluster_info']
    cluster_indices, cluster_num, k = cluster_info['cluster_indices'], cluster_info['cluster_num'], cluster_info['k']
    B, N = cluster_indices.shape
    device = cluster_indices.device
    rand_weights = torch.rand((B, N), device=device)
    cluster_ids = torch.arange(cluster_num, device=device).view(1, -1, 1)
    mask = (cluster_indices.unsqueeze(1) == cluster_ids)
    masked_weights = torch.where(mask, rand_weights.unsqueeze(1), torch.tensor(-float('inf'), device=device))
    fresh_indices = masked_weights.argmax(dim=2, keepdim=False)
    
    return fresh_indices

def select_fresh_indices_randomly(tokens, k):
    '''
    select k indices randomly (for comparison with ToCa)
    '''
    B, N, D = tokens.shape
    device = tokens.device
    fresh_indices = torch.randn((B, N), device=device).argsort(dim=1)[:, :k]
    return fresh_indices