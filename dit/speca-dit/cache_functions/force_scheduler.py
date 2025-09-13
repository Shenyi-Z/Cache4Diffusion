import torch
def force_scheduler(cache_dic, current):
    '''
    Force Activation Cycle Scheduler
    '''
    cache_dic['cal_threshold'] = cache_dic['interval']
