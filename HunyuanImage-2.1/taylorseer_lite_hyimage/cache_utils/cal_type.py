def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''

    first_steps = (current['step'] < cache_dic['first_enhance'])

    if (first_steps) or (cache_dic['cache_counter'] == cache_dic['cache_interval'] - 1 ):
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
    
    elif (cache_dic['taylor_cache']):
        cache_dic['cache_counter'] += 1
        current['type'] = 'Taylor'
    else:
        raise ValueError("Unsupported calculation type")