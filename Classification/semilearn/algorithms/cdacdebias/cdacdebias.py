# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, DebiasSamplingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.hooks.debias_sampling import get_append_type
import torch

import torch.nn.functional as F

from .utils import MmeBuildHook, advbce_unlabeled, sigmoid_rampup


@ALGORITHMS.register('cdacdebias')
class CDACDebias(AlgorithmBase):
    """
        Cross-Domain Adaptive Clustering (https://arxiv.org/abs/2104.09415).

        Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, cdac_specific_resnet=args.cdac_specific_resnet, reliable_rate=args.reliable_rate, num_append=args.num_append)
        self.append_type = get_append_type(args.append_type)
    
    def init(self, T, p_cutoff, hard_label=True, cdac_specific_resnet=False, reliable_rate=0.6, num_append=3):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.cdac_specific_resnet = cdac_specific_resnet
        self.reliable_rate = reliable_rate
        self.num_append = num_append

    def set_hooks(self):
        self.register_hook(DebiasSamplingHook(), "DebiasSamplingHook")
        self.register_hook(MmeBuildHook(), "MmeBuildHook")
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, idx_lb, x_lb, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1):
        num_lb_origin = y_lb.shape[0]
        x_lb, y_lb, y_lb_gt = self.call_hook(self.append_type, "DebiasSamplingHook", idx_lb, x_lb, y_lb, self.num_append)
        num_lb = y_lb.shape[0]
        
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1))
                feats = self.model(inputs, only_feat=True)
                feats_x_lb = feats[:num_lb]
                feats_x_ulb_w, feats_x_ulb_s, feats_x_ulb_s_1= feats[num_lb:].chunk(3)

                logits_x_lb = self.model(feats_x_lb, only_fc=True, reverse=False)
                logits_x_ulb_w = self.model(feats_x_ulb_w, only_fc=True, reverse=False)
                logits_x_ulb_s = self.model(feats_x_ulb_s, only_fc=True, reverse=False)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s_0)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':[feats_x_ulb_s, feats_x_ulb_s_1]}

            if self.args.num_labels == 0:
                sup_loss = torch.tensor(0.).cuda(self.args.gpu)
            else:
                ce_loss = self.ce_loss(logits_x_lb, y_lb, reduction='none')
                sup_loss = ce_loss[:num_lb_origin].mean() + ce_loss[num_lb_origin:].mean() * 1/self.num_append
            
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            pl_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)
            
            # consistency loss
            if self.cdac_specific_resnet:
                # for fair comparison, all algorithms use the same architecture. 
                # unless cdac-specific resnet is used, this loss degrades performance.
                # refer to: https://github.com/chu0802/SLA/blob/main/src/model.py#L78
                logits_x_ulb_s_1 = self.model(feats_x_ulb_s_1, only_fc=True, reverse=False)
                probs_x_ulb_s = self.compute_prob(logits_x_ulb_s)
                probs_x_ulb_s_1 = self.compute_prob(logits_x_ulb_s_1)
                w_cons = 30 * sigmoid_rampup(self.it, 2000)
                con_loss = w_cons * F.mse_loss(probs_x_ulb_s, probs_x_ulb_s_1)
            else:
                con_loss = torch.zeros(1).cuda(self.gpu)
            
            # compute cdac loss
            logits_x_ulb_w2 = self.model(feats_x_ulb_w, only_fc=True, reverse=True)
            logits_x_ulb_s2 = self.model(feats_x_ulb_s, only_fc=True, reverse=True)
            probs_x_ulb_w2 = self.compute_prob(logits_x_ulb_w2) 
            probs_x_ulb_s2 = self.compute_prob(logits_x_ulb_s2)
            aac_loss = advbce_unlabeled(target=None, f=feats_x_ulb_w, prob=probs_x_ulb_w2, prob1=probs_x_ulb_s2)

            total_loss = sup_loss + self.lambda_u * pl_loss + aac_loss + con_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         pl_loss=pl_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item(),
                                         aac_loss=aac_loss.item())
        return out_dict, log_dict
    

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--cdac_specific_resnet', str2bool, False),
            SSL_Argument('--reliable_rate', float, 0.4),
            SSL_Argument('--num_append', int, 3),
            SSL_Argument('--append_type', str, 'random'), # kmeans, random, opposite
        ]