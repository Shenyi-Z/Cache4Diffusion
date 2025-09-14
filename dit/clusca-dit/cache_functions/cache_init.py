def cache_init(model_kwargs, num_steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache[-1]={}

    for j in range(28):
        cache[-1][j] = {}
    for i in range(num_steps):
        cache[i]={}
        for j in range(28):
            cache[i][j] = {}

    cache_dic['cache']                = cache
    cache_dic['flops']                = 0.0
    cache_dic['interval']             = model_kwargs['interval']
    cache_dic['max_order']            = model_kwargs['max_order']
    cache_dic['test_FLOPs']           = model_kwargs['test_FLOPs']
    cache_dic['first_enhance']        = 2
    cache_dic['cache_counter']        = 0

    current = {}
    current['num_steps'] = num_steps
    current['activated_steps'] = [49]

    cache_dic['enable_clusca']        = model_kwargs.get('cluster_num', 0) > 0

    # ClusCa parameters
    if cache_dic['enable_clusca']:
        cache_dic['cluster_num']              = model_kwargs['cluster_num']
        cache_dic['k']                        = model_kwargs['k']
        cache_dic['propagation_ratio']        = model_kwargs['propagation_ratio']

        cache_dic['cluster_info'] = {}

    return cache_dic, current
    