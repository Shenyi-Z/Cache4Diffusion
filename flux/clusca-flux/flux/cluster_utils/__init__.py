import torch
from .Kmeans import Kmeans
import copy

def cluster_scheduler(cache_dic, current):
    return cache_dic['cluster_num'], cache_dic['topk']

def get_cluster_info(X, cache_dic, current):
    cluster_num, k = cluster_scheduler(cache_dic, current)
    cluster_info = copy.deepcopy(cache_dic['cluster_info'][current['stream']][current['module']])
    cache_centroids = cluster_info.get('centroids', None)
    cluster_indices, cache_centroids = Kmeans(n_clusters=cluster_num, init='random').fit(X, cache_centroids)
    cluster_info['cluster_num'] = cluster_num
    cluster_info['topk'] = k
    cluster_info['cluster_indices'] = cluster_indices
    cluster_info['centroids'] = cache_centroids
    cache_dic['cluster_info'][current['stream']][current['module']] = copy.deepcopy(cluster_info)

def construct_consecutive_cluster_info(X, cache_dic, current):
    '''
    construct consecutive cluster indices, every N//cluster_num tokens are grouped into one cluster
    '''
    cluster_num, k = cluster_scheduler(cache_dic, current)
    B, N, D = X.shape
    device = X.device
    segment_length = N // cluster_num
    cluster_indices = torch.arange(cluster_num, dtype=torch.long, device=device).repeat_interleave(segment_length)
    cluster_indices = cluster_indices.unsqueeze(0).expand(B, -1)
    cache_dic['cluster_info']['cluster_num'] = cluster_num
    cache_dic['cluster_info']['topk'] = k
    cache_dic['cluster_info']['cluster_indices'] = cluster_indices
    
def random_cluster_indices(X, cache_dic, current):
    '''
    randomly group the tokens into cluster_num groups(for ablation study)
    '''
    cluster_num, k = cluster_scheduler(cache_dic, current)
    B, N, D = X.shape
    device = X.device
    cluster_indices = torch.randint(0, cluster_num, (B, N), device=device)
    cache_dic['cluster_info']['cluster_indices'] = cluster_indices
    cache_dic['cluster_info']['cluster_num'] = cluster_num
    cache_dic['cluster_info']['topk'] = k
    