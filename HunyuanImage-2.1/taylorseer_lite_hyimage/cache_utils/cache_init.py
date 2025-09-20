
def cache_init(num_steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache[-1]={}

    cache[-1]['double_stream']={}
    cache[-1]['single_stream']={}
    cache_dic['cache_counter'] = 0

    cache[-1]['final'] = {}
    cache[-1]['final']['final'] = {}
    cache[-1]['final']['final']['final'] = {}

    cache_dic['cache'] = cache

    mode = 'fast' #['fast', 'mid', 'detailed'], seems no visible quality drop, so just use fast. Choices are set for some usage.

    if mode == 'fast': 
        cache_dic['cache_interval'] = 5
        cache_dic['max_order'] = 2
        cache_dic['first_enhance'] = 3

    elif mode == 'mid':
        cache_dic['cache_interval'] = 4
        cache_dic['max_order'] = 2
        cache_dic['first_enhance'] = 3
        
    elif mode == 'detailed':
        cache_dic['cache_interval'] = 3
        cache_dic['max_order'] = 2
        cache_dic['first_enhance'] = 3
    
    cache_dic['taylor_cache'] = True
    
    current = {}
    current['activated_steps'] = [0]

    current['step'] = 0
    current['num_steps'] = num_steps

    return cache_dic, current
