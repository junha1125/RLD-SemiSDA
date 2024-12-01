# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch.nn import functional as F

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from .utils import HistorySaveHook


@ALGORITHMS.register('fixmatchsaver')
class FixMatchSaver(AlgorithmBase):

    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

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
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, gamma=args.gamma)
    
    def init(self, T, p_cutoff, hard_label=True, gamma=2.):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.gamma = gamma
    
    def set_hooks(self):
        self.register_hook(HistorySaveHook(), "HistorySaveHook")
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
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            if self.args.num_labels == 0:
                sup_loss = torch.tensor(0.).cuda(self.args.gpu)
            else:
                sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
                # sup_loss = self.ggl_loss(logits_x_lb, y_lb)
            
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--gamma', float, 2.),
        ]
    
    def ggl_loss(self, logits, targets):
        """
        gradual guided learning
        Args:
            logits (Tensor, shape: [batch size, num_class]): logits from the model
            targets (Tensor, shape: [batch size]): GT labels
        """
        prob_x_lb = self.compute_prob(logits)
        prob_gt = prob_x_lb[torch.arange(prob_x_lb.shape[0]), targets].detach()
        prob_pr = prob_x_lb.max(dim=-1)[0]
        idx_incorr = prob_x_lb.max(dim=-1)[1] != targets
        ce_loss = (prob_gt**self.gamma) * self.ce_loss(logits, targets, reduction='none')
        mse_loss = (1-prob_gt[idx_incorr])**self.gamma * (prob_pr[idx_incorr])**2
        # max_ce_loss = (1-prob_gt[idx_incorr])**self.gamma * torch.log(prob_pr[idx_incorr])
        loss = ce_loss.mean() + mse_loss.mean() if idx_incorr.sum() else ce_loss.mean()
        return loss
    
        """
        prob_x_lb = self.compute_prob(logits)
        # prob_x_lb.shape = torch.Size([4, 126])
        prob_gt = prob_x_lb[torch.arange(prob_x_lb.shape[0]), targets].detach()
        # [0.0348, 0.7579, 0.7975, 0.0026]
        prob_pr = prob_x_lb.max(dim=-1)[0]
        # [0.1315, 0.7579, 0.7975, 0.1214]
        idx_incorr = prob_x_lb.max(dim=-1)[1] != targets
        # [ True, False, False,  True]
        ce_loss = (prob_gt**self.gamma) * self.ce_loss(logits, targets, reduction='none')
        # [0.1168, 0.2101, 0.1805, 0.0157] = [0.0348, 0.7579, 0.7975, 0.0026] * [3.3587, 0.2772, 0.2263, 5.9376]
        # [0.1168, 0.2101, 0.1805, 0.0157]
        mse_loss = (1-prob_gt[idx_incorr])**self.gamma * (prob_pr[idx_incorr])**2
        # [0.0167, 0.0147] = [0.9652, 0.9974] * [0.1315, 0.1214] ** 2
        # max_ce_loss = (1-prob_gt[idx_incorr])**self.gamma * torch.log(prob_pr[idx_incorr])
        # # [-1.9583, -2.1029] = [0.9652, 0.9974] * [-2.0288, -2.1085] ** 2
        loss = ce_loss.mean() + mse_loss.mean() if idx_incorr.sum() else ce_loss.mean()
        return loss
        """

        
        