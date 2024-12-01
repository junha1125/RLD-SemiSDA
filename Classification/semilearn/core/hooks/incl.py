# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from .hook import Hook

try:
    import incl
    INCL_IMPORTED = True
except :
    INCL_IMPORTED = False



class INCLHook(Hook):
    """
    Wandb Hook
    """

    def __init__(self):
        super().__init__()
        # if you want to log only your desired keys, you can specify them here
        # self.log_key_list = ['train/sup_loss', 'train/unsup_loss', 'train/total_loss', 'train/util_ratio', 
        #                      'train/run_time', 'train/prefetch_time', 'lr',
        #                      'eval/top-1-acc', 'eval/precision', 'eval/recall', 'eval/F1',
        #                      'eval_src/top-1-acc']
        # self.eval_log_key_list = ['eval/top-1-acc', 'eval_src/top-1-acc']

    def before_run(self, algorithm):
        if not INCL_IMPORTED: algorithm.print_fn("INCL is not installed. Rerun without --use_incl"); return 
        args_dict = vars(algorithm.args)
        incl.config.init(args_dict)

        log_dict = {}
        for key, item in algorithm.log_dict.items():
            if not isinstance(item, str): # if key in self.log_key_list:
                log_dict[key] = item
        incl.log(log_dict, step=algorithm.it)


    def after_train_step(self, algorithm):
        if not INCL_IMPORTED: return 
        if self.every_n_iters(algorithm, algorithm.num_log_iter):
            log_dict = {}
            for key, item in algorithm.log_dict.items() :
                if not isinstance(item, str): #  if key in self.log_key_list:
                    log_dict[key] = item
            incl.log(log_dict, step=algorithm.it)
    
        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            incl.log({'eval/best-acc': algorithm.best_eval_acc}, step=algorithm.it)
            log_dict = {}
            for key, item in algorithm.log_dict.items():
                if not isinstance(item, str): #  if key in self.eval_log_key_list:
                    log_dict[key] = item
            incl.log(log_dict, step=algorithm.it)
    