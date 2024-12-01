# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument
import torch
import torch.nn.functional as F

from .utils import MmeBuildHook


@ALGORITHMS.register('mme')
class MME(AlgorithmBase):
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

    def set_hooks(self):
        self.register_hook(MmeBuildHook(), "MmeBuildHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w):
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
                sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            probs_x_ulb = F.softmax(logits_x_ulb, dim=1)
            lambda_u = 0.1
            unsup_loss = lambda_u * torch.mean(torch.sum(probs_x_ulb * (torch.log(probs_x_ulb + 1e-5)), dim=1))

            total_loss = sup_loss + unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(),)
        return out_dict, log_dict
