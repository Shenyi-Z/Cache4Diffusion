def cache_init(num_steps, model_kwargs=None):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1] = {}
    cache_index[-1] = {}
    cache_index['layer_index'] = {}
    cache[-1]['double_stream'] = {}
    cache[-1]['single_stream'] = {}

    for j in range(20):
        cache[-1]['double_stream'][j] = {}
        cache_index[-1][j] = {}

    for j in range(40):
        cache[-1]['single_stream'][j] = {}
        cache_index[-1][j] = {}

    cache_dic['taylor_cache'] = False
    cache_dic['test_FLOPs'] = False

    mode = 'Taylor'
    if mode == 'original':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 1
        cache_dic['error_metric'] = 'relative_l1' 
        
    elif mode == 'Taylor':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['max_order'] = 1
        cache_dic['taylor_cache'] = True
        cache_dic['first_enhance'] = 1
        cache_dic['taylor_step_counter'] = 0
        cache_dic['check'] = False
        cache_dic['base_threshold'] = 1
        cache_dic['decay_rate'] = 0.1
        cache_dic['min_taylor_steps'] = 4
        cache_dic['max_taylor_steps'] = 12
        cache_dic['error_metric'] = 'relative_l1'  

    current = {}
    current['num_steps'] = num_steps
    current['activated_steps'] = [0]
    current['last_type'] = 'None'
    current['last_layer_error'] = 0.0

    return cache_dic, current
