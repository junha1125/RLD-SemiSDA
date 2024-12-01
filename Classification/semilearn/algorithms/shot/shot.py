# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument
import torch

from .utils import ObtainClusterLabel, compute_entropy


@ALGORITHMS.register('shot')
class SHOT(AlgorithmBase):
    """
        SHOT algorithm (https://arxiv.org/abs/2002.08546).

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
        self.init(distance=args.distance, threshold=args.threshold, cls_par=args.cls_par, ent_par=args.ent_par)
    def init(self, distance, threshold, cls_par, ent_par):
        self.distance = distance
        self.threshold = threshold
        self.cls_par = cls_par
        self.ent_par = ent_par

    def set_hooks(self):
        self.register_hook(ObtainClusterLabel(), "ObtainClusterLabel")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w):
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
                sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            pred = self.call_hook("obtain_current_label","ObtainClusterLabel", tar_idx=idx_ulb)
            classifier_loss = torch.nn.CrossEntropyLoss()(logits_x_ulb, pred)
            classifier_loss *= self.cls_par

            if self.it < self.num_iter_per_epoch and self.args.dataset == "visda":
                classifier_loss *= 0

            softmax_out = torch.nn.Softmax(dim=1)(logits_x_ulb)
            entropy_loss = torch.mean(compute_entropy(softmax_out))

            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
            entropy_loss -= gentropy_loss
            entropy_loss *= self.ent_par

            total_loss = sup_loss + classifier_loss + entropy_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         classifier_loss=classifier_loss.item(), 
                                         entropy_loss=entropy_loss.item(),
                                         total_loss=total_loss.item(), )
        return out_dict, log_dict
    
    def set_optimizer(self):
        optimizer, scheduler = super().set_optimizer()
        if 'resnet' in self.args.net:
            # if resnet is used, we need to pop the last layer following the original implementation
            for param in self.model.fc.parameters():
                param.requires_grad = False
            last_layer_shape = optimizer.param_groups[0]['params'][-1].shape 
            assert optimizer.param_groups[0]['name'] == 'classifier' and last_layer_shape[0] == self.args.num_classes and\
                last_layer_shape[1] == self.model.output_dim, "Do you really pop the last layer?"
            optimizer.param_groups[0]['params'].pop(-1)
        else:
            # leave log 'if architecture is not resnet, we do not freeze the last layer'
            self.print_fn("[SHOT algorithm] if architecture is not resnet, we do not freeze the last layer")
        return optimizer, scheduler

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['mem_label'] = self.hooks_dict['ObtainClusterLabel'].mem_label.cpu()
        # save_dict['ent_threshold'] = self.hooks_dict['ObtainClusterLabel'].ent_threshold # for ODA setting
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        if 'mem_label' not in checkpoint:
            self.print_fn("additional parameter not found")
        else:
            self.hooks_dict['ObtainClusterLabel'].mem_label = checkpoint['mem_label'].cuda(self.args.gpu)
            # self.hooks_dict['ObtainClusterLabel'].ent_threshold = checkpoint['ent_threshold'] # for ODA setting
            self.print_fn("additional parameter loaded")
        return checkpoint


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--distance', str, "cosine"),
            SSL_Argument('--threshold', float, 0.0),
            SSL_Argument('--cls_par', float, 0.3),
            SSL_Argument('--ent_par', float, 1.0),
        ]