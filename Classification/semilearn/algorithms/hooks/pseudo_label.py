# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.core.hooks import Hook
from semilearn.algorithms.utils import smooth_targets

class PseudoLabelingHook(Hook):
    """
    Pseudo Labeling Hook
    """
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def gen_ulb_targets(self, 
                        algorithm, 
                        logits, 
                        use_hard_label=True, 
                        T=1.0,
                        softmax=True, # whether to compute softmax for logits, input must be logits
                        label_smoothing=0.0):
        
        """
        generate pseudo-labels from logits/probs

        Args:
            algorithm: base algorithm
            logits: logits (or probs, need to set softmax to False)
            use_hard_label: flag of using hard labels instead of soft labels
            T: temperature parameters
            softmax: flag of using softmax on logits
            label_smoothing: label_smoothing parameter
        """

        logits = logits.detach()
        if use_hard_label:
            # return hard label directly
            pseudo_label = torch.argmax(logits, dim=-1)
            if label_smoothing:
                pseudo_label = smooth_targets(logits, pseudo_label, label_smoothing)
            return pseudo_label
        
        # return soft label
        if softmax:
            # pseudo_label = torch.softmax(logits / T, dim=-1)
            pseudo_label = algorithm.compute_prob(logits / T)
        else:
            # inputs logits converted to probabilities already
            pseudo_label = logits
        return pseudo_label
        

    @torch.no_grad()
    def gen_ulb_targets_based_history(self, 
                        algorithm, 
                        logits,
                        history_label,
                        label_smoothing=0.1):
        
        """
        generate smoothed pseudo-labels from logits/probs based on history labels

        Args:
            algorithm: base algorithm
            logits: logits (or probs, need to set softmax to False)
            history_label: label history of unlabeled data, shape=[batch_size, num_history]
            label_smoothing: label_smoothing parameter
        """
        pseudo_label = torch.zeros_like(logits)
        for idx, one_data_history in enumerate(history_label):
            main_label = one_data_history[-1]
            sub_label = torch.unique(one_data_history)
            if sub_label.shape[0] <= 1:
                pass
            else:
                pseudo_label[idx][sub_label] = label_smoothing / (sub_label.shape[0] - 1)
            pseudo_label[idx][main_label] = 1-label_smoothing
        return pseudo_label
