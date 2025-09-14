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
    cache_dic['max_order']            = model_kwargs['max_order']
    cache_dic['test_FLOPs']           = model_kwargs['test_FLOPs']
    cache_dic['first_enhance']        = 3
    cache_dic['cache_counter']        = 0
    cache_dic['taylor_step_counter']  = 0
    cache_dic['check']                = False
    cache_dic['base_threshold']       = model_kwargs['base_threshold']
    cache_dic['decay_rate']           = model_kwargs['decay_rate']
    cache_dic['min_taylor_steps']     = model_kwargs['min_taylor_steps']
    cache_dic['max_taylor_steps']     = model_kwargs['max_taylor_steps']
    cache_dic['error_metric']         = model_kwargs['error_metric']
    
    current = {}
    current['last_layer_error']       = 0.0
    current['num_steps']              = num_steps
    current['activated_steps']        = [num_steps-1]  # Start from the last step
    current['last_type']              = 'None'

    return cache_dic, current
