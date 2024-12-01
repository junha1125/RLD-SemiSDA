# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import DebiasSamplingHook
from semilearn.algorithms.utils import SSL_Argument
from semilearn.algorithms.hooks.debias_sampling import get_append_type
import torch
import torch.nn.functional as F

from .utils import MmeBuildHook


@ALGORITHMS.register('mmedebias')
class MMEDebias(AlgorithmBase):
    """
        Pseudo Label algorithm (https://arxiv.org/abs/1908.02983).

        Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.init(reliable_rate=args.reliable_rate, num_append=args.num_append)
        self.append_type = get_append_type(args.append_type)

    def init(self, reliable_rate=0.6, num_append=3):
        self.reliable_rate = reliable_rate
        self.num_append = num_append

    def set_hooks(self):
        self.register_hook(DebiasSamplingHook(), "DebiasSamplingHook")
        self.register_hook(MmeBuildHook(), "MmeBuildHook")
        super().set_hooks()

    def train_step(self, idx_lb, x_lb, y_lb, x_ulb_w):
        num_lb_origin = y_lb.shape[0]
        x_lb, y_lb, y_lb_gt = self.call_hook(self.append_type, "DebiasSamplingHook", idx_lb, x_lb, y_lb, self.num_append)
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w))
                feats = self.model(inputs, only_feat=True)
                feats_x_lb = feats[:num_lb]
                feats_x_ulb  = feats[num_lb:]
                logits_x_lb = self.model(feats_x_lb, only_fc=True, reverse=False)
                logits_x_ulb = self.model(feats_x_ulb, only_fc=True, reverse=True)
            else:
                raise NotImplementedError

            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb}

            if self.args.num_labels == 0:
                sup_loss = torch.tensor(0.).cuda(self.args.gpu)
            else:
                ce_loss = self.ce_loss(logits_x_lb, y_lb, reduction='none')
                sup_loss = ce_loss[:num_lb_origin].mean() + ce_loss[num_lb_origin:].mean() * 1/self.num_append

            probs_x_ulb = F.softmax(logits_x_ulb, dim=1)
            lambda_u = 0.1
            unsup_loss = lambda_u * torch.mean(torch.sum(probs_x_ulb * (torch.log(probs_x_ulb + 1e-5)), dim=1))

            total_loss = sup_loss + unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(),)
        return out_dict, log_dict
    
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--reliable_rate', float, 0.4),
            SSL_Argument('--num_append', int, 3),
            SSL_Argument('--append_type', str, 'random'), # kmeans, random, opposite
        ]
