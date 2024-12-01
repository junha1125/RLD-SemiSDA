# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool, smoothed_cross_entropy
from functools import partial


@ALGORITHMS.register('simplefinetune')
class SimpleFinetune(AlgorithmBase):

    """
        SimpleFinetune algorithm.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - supervised_type (`str`, *optional*, default to `celoss`):
                Supervised loss type. Options: `zero`, `celoss`, `smooth_celoss`.
            - unsupervised_type (`str`, *optional*, default to `celoss`):
                Unsupervised loss type. Options: `zero`, `celoss`, `smooth_celoss`, `consistency`.

    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, \
                   supervised_type=args.supervised_type, unsupervised_type=args.unsupervised_type)
        self.smooth_ce_loss = partial(smoothed_cross_entropy, num_classes=self.num_classes)
    
    def init(self, T, p_cutoff, hard_label, supervised_type, unsupervised_type):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.supervised_type = supervised_type
        self.unsupervised_type = unsupervised_type
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                if self.unsupervised_type != 'zero':
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
                    outs_x_ulb_s = self.model(x_ulb_s)
                    logits_x_ulb_s = outs_x_ulb_s['logits']
                    feats_x_ulb_s = outs_x_ulb_s['feat']
            feat_dict = {'x_lb':feats_x_lb,} # 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}




            # supervised loss
            if self.supervised_type == 'celoss':
                sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            elif self.supervised_type == 'smooth_celoss':
                sup_loss = self.smooth_ce_loss(logits_x_lb, y_lb)
            elif self.supervised_type == 'zero':
                # get torch tensor float zero
                sup_loss = torch.tensor(0.).cuda(self.gpu)
            else:
                raise Exception('Unknown supervised loss type: {} (Options: zero, celoss, smooth_celoss)'.format(self.supervised_type))

            # unsupervised loss
            # TODO: clean code
            if self.unsupervised_type != 'zero':
                probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
            if self.unsupervised_type == 'celoss':
                pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                            logits=probs_x_ulb_w,
                                            use_hard_label=self.use_hard_label,
                                            T=self.T,
                                            softmax=False)
                unsup_loss = self.consistency_loss(probs_x_ulb_w,
                                                pseudo_label,
                                                'ce',
                                                mask=mask)
            elif self.unsupervised_type == 'smooth_celoss':
                pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                            logits=probs_x_ulb_w,
                                            use_hard_label=self.use_hard_label,
                                            T=self.T,
                                            softmax=False,
                                            label_smoothing=0.1)                            # main differnece
                unsup_loss = self.consistency_loss(probs_x_ulb_w,
                                                pseudo_label,
                                                'ce',
                                                mask=mask)
            elif self.unsupervised_type == 'consistency':
                pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                            logits=probs_x_ulb_w,
                                            use_hard_label=self.use_hard_label,
                                            T=self.T,
                                            softmax=False)
                unsup_loss = self.consistency_loss(logits_x_ulb_s,                          # main differnece
                                                pseudo_label,
                                                'ce',
                                                mask=mask)
            elif self.unsupervised_type == 'zero':
                unsup_loss = torch.tensor(0.).cuda(self.gpu)
            else:
                raise Exception('Unknown unsupervised loss type: {} (Options: zero, celoss, smooth_celoss, consistency)'.format(self.unsupervised_type))

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(),)
        return out_dict, log_dict
        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--supervised_type', str, 'Not defined'),    
            SSL_Argument('--unsupervised_type', str, 'Not defined'),  
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
