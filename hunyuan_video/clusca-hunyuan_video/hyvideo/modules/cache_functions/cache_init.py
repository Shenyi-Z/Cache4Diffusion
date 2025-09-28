def cache_init(num_steps, model_kwargs=None):   
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

    for j in range(20):
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

    for j in range(40):
        cache[-1]['single_stream'][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1]['single_stream'][j] = {}
        cache_dic['attn_map'][-1]['single_stream'][j]['total'] = {}

        cache_dic['k-norm'][-1]['single_stream'][j] = {}
        cache_dic['k-norm'][-1]['single_stream'][j]['total'] = {}

        cache_dic['v-norm'][-1]['single_stream'][j] = {}
        cache_dic['v-norm'][-1]['single_stream'][j]['total'] = {}

    cache_dic['taylor_cache'] = False
    cache_dic['duca']  = False
    cache_dic['test_FLOPs'] = False

    cache_dic['mode'] = model_kwargs['mode']
    mode = cache_dic['mode']
    
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
        cache_dic['first_enhance'] = 1
        
    elif mode == 'ToCa':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.10
        cache_dic['fresh_threshold'] = 5
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 1
        cache_dic['duca'] = False

    elif mode == 'DuCa':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.10
        cache_dic['fresh_threshold'] = 5
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 1
        cache_dic['duca'] = True

    elif mode == 'Taylor':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 5
        cache_dic['max_order'] = 1
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['taylor_cache'] = True
        cache_dic['first_enhance'] = 1
    
    elif mode == 'ClusCa':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.1
        cache_dic['fresh_threshold'] = model_kwargs['fresh_threshold']
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['taylor_cache'] = True
        cache_dic['max_order'] = model_kwargs['max_order']
        cache_dic['first_enhance'] = 1 
        cache_dic['cluster_info'] = {}
        cache_dic['cluster_num'] = model_kwargs['cluster_num']
        cache_dic['T-wise'] = model_kwargs['T-wise']
    

        cluster_info_dict = {}
        cluster_info_dict['cluster_indices'] = None
        cluster_info_dict['centroids'] = None
        
        cache_dic['cluster_info']['double_stream'] = {}
        cache_dic['cluster_info']['single_stream'] = {}
        cache_dic['cluster_info']['double_stream']['img_mlp'] = cluster_info_dict
        cache_dic['cluster_info']['double_stream']['txt_mlp'] = cluster_info_dict
        
        cache_dic['k'] = model_kwargs['k']
        cache_dic['cluster_info']['cluster_indices'] = None
        cache_dic['cluster_info']['centroids'] = None
        cache_dic['propagation_ratio'] = model_kwargs['propagation_ratio']

    current = {}
    current['num_steps'] = num_steps
    current['activated_steps'] = [0]

    return cache_dic, current
