def cal_type(cache_dic, current):  
    min_taylor_steps = cache_dic['min_taylor_steps']
    max_taylor_steps = cache_dic['max_taylor_steps']

    if 'full_count' not in cache_dic:
        cache_dic['full_count'] = 0  

    if current['last_type'] == 'full':
        current['type'] = 'Taylor'
        cache_dic['taylor_step_counter'] = 1  
        cache_dic['check'] = False
        current['last_layer_error'] = None
    else:
        first_steps = (current['step'] > (current['num_steps'] - cache_dic['first_enhance'] - 1))
        reached_max_taylor = (cache_dic['taylor_step_counter'] >= max_taylor_steps)
        progress = (current['num_steps'] - current['step']) / current['num_steps']
        base_threshold = cache_dic['base_threshold']
        decay_rate = cache_dic['decay_rate']
        threshold = base_threshold * (decay_rate ** progress)
        threshold = max(threshold, 0.01) 

        if cache_dic['taylor_step_counter'] >= min_taylor_steps:
            cache_dic['check'] = True
        else:
            cache_dic['check'] = False

        error_too_large = current.get('last_layer_error') is not None and current.get('last_layer_error') > threshold

        if first_steps:
            current['type'] = 'full'
            cache_dic['taylor_step_counter'] = 0
            cache_dic['full_count'] += 1  
            
        elif reached_max_taylor:
            current['type'] = 'full'
            cache_dic['taylor_step_counter'] = 0
            cache_dic['full_count'] += 1  
            
        elif error_too_large and cache_dic['check']:
            current['type'] = 'full'
            cache_dic['taylor_step_counter'] = 0
            cache_dic['full_count'] += 1 
        elif cache_dic['taylor_step_counter'] < min_taylor_steps:
            current['type'] = 'Taylor'
            cache_dic['taylor_step_counter'] += 1
        else:
            current['type'] = 'Taylor'
            cache_dic['taylor_step_counter'] += 1

    current['last_type'] = current['type']

    if current['type'] == 'full':
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
    else:
        cache_dic['cache_counter'] += 1


