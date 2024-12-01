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

from .utils import ObtainNRCLabel


@ALGORITHMS.register('nrcdebias')
class NRCDebias(AlgorithmBase):
    """
        NRC algorithm (https://arxiv.org/abs/2110.04202).

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
        self.init(K=args.K, KK=args.KK, reliable_rate=args.reliable_rate, num_append=args.num_append)
        self.append_type = get_append_type(args.append_type)
    
    def init(self, K, KK, reliable_rate=0.6, num_append=3):
        self.K = K
        self.KK = KK
        self.reliable_rate = reliable_rate
        self.num_append = num_append

    def set_hooks(self):
        self.register_hook(DebiasSamplingHook(), "DebiasSamplingHook")
        self.register_hook(ObtainNRCLabel(), "ObtainNRCLabel")
        super().set_hooks()

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w):
        num_lb_origin = y_lb.shape[0]
        x_lb, y_lb, y_lb_gt = self.call_hook(self.append_type, "DebiasSamplingHook", idx_lb, x_lb, y_lb, self.num_append)
        num_lb = y_lb.shape[0]

        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb = outputs['logits'][num_lb:]
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb  = outputs['feat'][num_lb:]
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_w = self.model(x_ulb_w)
                logits_x_ulb = outs_x_ulb_w['logits']
                feats_x_ulb = outs_x_ulb_w['feat']

            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb}

            if self.args.num_labels == 0:
                sup_loss = torch.tensor(0.).cuda(self.args.gpu)
            else:
                ce_loss = self.ce_loss(logits_x_lb, y_lb, reduction='none')
                sup_loss = ce_loss[:num_lb_origin].mean() + ce_loss[num_lb_origin:].mean() * 1/self.num_append

            softmax_out = torch.nn.Softmax(dim=1)(logits_x_ulb)
            # TODO: why dont extract softmax_out and feats_x_ulb using ema_model?
            score_near, weight, score_near_kk, weight_kk = self.call_hook("obtain_neighbors", "ObtainNRCLabel", \
                                            tar_idx=idx_ulb, features_test=feats_x_ulb, softmax_out=softmax_out)
            
            # nn of nn
            # kl_div loss equals to dot product since we do not use log for score_near_kk
            output_re = softmax_out.unsqueeze(1).expand(-1, self.K * self.KK, -1)  # batch x C x 1
            const = torch.mean((F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) * weight_kk.cuda()).sum(1)) 
            loss_near_kk = torch.mean(const)

            # nn
            softmax_out_un = softmax_out.unsqueeze(1).expand(-1, self.K, -1)  # batch x K x C
            loss_near = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.cuda()).sum(1))

            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

            total_loss = sup_loss + loss_near_kk + loss_near + gentropy_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         loss_near_kk=loss_near_kk.item(), 
                                         loss_near=loss_near.item(),
                                         gentropy_loss=gentropy_loss.item(),
                                         total_loss=total_loss.item(), )
        return out_dict, log_dict


    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['fea_bank'] = self.hooks_dict['ObtainNRCLabel'].fea_bank.cpu()
        save_dict['score_bank'] = self.hooks_dict['ObtainNRCLabel'].score_bank.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        if 'fea_bank' not in checkpoint:
            self.print_fn("additional parameter not found")
        else:
            self.hooks_dict['ObtainNRCLabel'].fea_bank = checkpoint['fea_bank'].cuda(self.args.gpu)
            self.hooks_dict['ObtainNRCLabel'].score_bank = checkpoint['score_bank'].cuda(self.args.gpu)
            self.print_fn("additional parameter loaded")
        return checkpoint


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--K', int, 5),
            SSL_Argument('--KK', int, 5),
            SSL_Argument('--reliable_rate', float, 0.4),
            SSL_Argument('--num_append', int, 3),
            SSL_Argument('--append_type', str, 'random'), # kmeans, random, opposite
        ]