import os
from .hook import Hook

class RedefineDataHook(Hook):
    """
    Redefine dataset and dataloader
    """

    def before_run(self, algorithm):
        if algorithm.args.trg_eval_src or algorithm.args.save_feature:
            algorithm.print_fn("[RedefineDataHook] during fine-tuning, do additional evaluation on source domain")
            algorithm.redefine_data_for_src()
        if algorithm.args.negatively_biased_feedback:
            algorithm.print_fn("[RedefineDataHook] as much as possible, labeled dataset include samples predicted as false labels")
            algorithm.redefine_train_lb()