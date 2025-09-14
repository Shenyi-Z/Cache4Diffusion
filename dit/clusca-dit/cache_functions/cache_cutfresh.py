from .fresh_ratio_scheduler import fresh_ratio_scheduler
from .score_evaluate import score_evaluate
#from .token_merge import token_merge
import torch
from .select_fresh_tokens import select_fresh_indices_randomly, select_one_fresh_index_per_cluster
def cache_cutfresh(cache_dic, tokens, current):
    '''
    Cut fresh tokens from the input tokens and update the cache counter.
    
    cache_dic: dict, the cache dictionary containing cache(main extra memory cost), indices and some other information.
    tokens: torch.Tensor, the input tokens to be cut.
    current: dict, the current step, layer, and module information. Particularly convenient for debugging.
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']
    
    
    fresh_indices = select_one_fresh_index_per_cluster(cache_dic, current)

    # select the fresh tokens out
    fresh_indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    if module in ['mlp', 'attn']:
        # cut out the fresh tokens
        fresh_tokens = torch.gather(input = tokens, dim = 1, index = fresh_indices_expand)

        return fresh_indices, fresh_tokens
    
    else:
        # no need for this branch hhh.
        raise ValueError("Unrecognized module?", module)