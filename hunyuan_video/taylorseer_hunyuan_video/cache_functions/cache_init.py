
'''
In config.json, there is only "Name" attribute and nothing like "num_single_layers", "num_double_layers",
so we refer to the number of layers in the technique report of HunyuanVideo 


'''

# num_double_layers = 20
# num_single_layers = 40


from diffusers.models import HunyuanVideoTransformer3DModel
def cache_init(self: HunyuanVideoTransformer3DModel, ):   
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

    cache[-1]['double_stream']={}
    cache[-1]['single_stream']={}
    cache_dic['cache_counter'] = 0

    # for j in range(20):
    for j in range(self.config.num_layers):
        cache[-1]['double_stream'][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1]['double_stream'][j] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['total'] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['txt_mlp'] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['img_mlp'] = {}

    #for j in range(40):
    for j in range(self.config.num_single_layers):
        cache[-1]['single_stream'][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1]['single_stream'][j] = {}
        cache_dic['attn_map'][-1]['single_stream'][j]['total'] = {}

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
    
    elif mode == 'Taylor':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 6
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['taylor_cache'] = True
        cache_dic['max_order'] = 1
        cache_dic['first_enhance'] = 3

    current = {}
    current['activated_steps'] = [0]
    current['step'] = 0
    current['num_steps'] = self.num_steps
    # print("self.num_steps:", self.num_steps)  # check if self.num_steps exists while the name in args is infer_steps, the result is yes.

    return cache_dic, current
