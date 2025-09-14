def cache_init(timesteps, model_kwargs=None):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}
    cache_dic['attn_map'] = {}
    cache_dic['attn_map'][-1] = {}
    cache_dic['attn_map'][-1]['double_stream'] = {}
    cache_dic['attn_map'][-1]['single_stream'] = {}

    cache_dic['k-norm'] = {}
    cache_dic['k-norm'][-1] = {}
    cache_dic['k-norm'][-1]['double_stream'] = {}
    cache_dic['k-norm'][-1]['single_stream'] = {}

    cache_dic['v-norm'] = {}
    cache_dic['v-norm'][-1] = {}
    cache_dic['v-norm'][-1]['double_stream'] = {}
    cache_dic['v-norm'][-1]['single_stream'] = {}

    cache_dic['cross_attn_map'] = {}
    cache_dic['cross_attn_map'][-1] = {}
    cache[-1]['double_stream']={}
    cache[-1]['single_stream']={}
    cache_dic['cache_counter'] = 0

    for j in range(19):
        cache[-1]['double_stream'][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1]['double_stream'][j] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['total'] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['txt_mlp'] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['img_mlp'] = {}
        
        cache_dic['k-norm'][-1]['double_stream'][j] = {}
        cache_dic['k-norm'][-1]['double_stream'][j]['txt_mlp'] = {}
        cache_dic['k-norm'][-1]['double_stream'][j]['img_mlp'] = {}

        cache_dic['v-norm'][-1]['double_stream'][j] = {}
        cache_dic['v-norm'][-1]['double_stream'][j]['txt_mlp'] = {}
        cache_dic['v-norm'][-1]['double_stream'][j]['img_mlp'] = {}

    for j in range(38):
        cache[-1]['single_stream'][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1]['single_stream'][j] = {}
        cache_dic['attn_map'][-1]['single_stream'][j]['total'] = {}

        cache_dic['k-norm'][-1]['single_stream'][j] = {}
        cache_dic['k-norm'][-1]['single_stream'][j]['total'] = {}

        cache_dic['v-norm'][-1]['single_stream'][j] = {}
        cache_dic['v-norm'][-1]['single_stream'][j]['total'] = {}

    cache_dic['taylor_cache'] = False
    cache_dic['Delta-DiT'] = False

    mode = 'Taylor'

    if mode == 'original':
        cache_dic['cache_type'] = 'random' 
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa'
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 1
        cache_dic['force_fresh'] = 'global'
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 3
        
    elif mode == 'ToCa':
        cache_dic['cache_type'] = 'attention'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.1
        cache_dic['fresh_threshold'] = 5
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 3
    
    elif mode == 'Taylor':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 0
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['taylor_cache'] = True
        cache_dic['max_order'] =  2
        cache_dic['first_enhance'] = 3
        cache_dic['taylor_step_counter']  = 0
        cache_dic['check']                = False
        
        cache_dic['base_threshold']   = model_kwargs['base_threshold']
        cache_dic['decay_rate']   = model_kwargs['decay_rate']
        cache_dic['min_taylor_steps']   = model_kwargs['min_taylor_steps']
        cache_dic['max_taylor_steps']   = model_kwargs['max_taylor_steps']
        cache_dic['error_metric'] = 'relative_l1'  


        cache_dic['error_metric'] = 'relative_l1'  
 
        

    elif mode == 'Delta':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 3
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['Delta-DiT'] = True
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 1

    current = {}
    current['final_time'] = timesteps[-2]
    current['activated_steps'] = [0]
    current['last_type']              = 'None'
    current['last_layer_error']       = 0.0
    return cache_dic, current
