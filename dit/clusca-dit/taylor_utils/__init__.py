from typing import Dict
import torch
import math

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    # difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['layer']][current['module']].get(i, None) is not None) and (current['step'] < (current['num_steps'] - cache_dic['first_enhance'] + 1)):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['layer']][current['module']][i]) / difference_distance
        else:
            break
    
    cache_dic['cache'][-1][current['layer']][current['module']] = updated_taylor_factors

def taylor_formula(module_dict: Dict, distance:int) -> torch.Tensor: 
    """
    Compute Taylor expansion error.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    #x = current['step'] - current['activated_steps'][-1]
    # x = current['t'] - current['activated_times'][-1]
    output = 0

    for i in range(len(module_dict)):
        output += (1 / math.factorial(i)) * module_dict[i] * (distance ** i)
    
    return output

def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and expand storage for different-order derivatives.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    if current['step'] == (current['num_steps'] - 1):
        cache_dic['cache'][-1][current['layer']][current['module']] = {}
