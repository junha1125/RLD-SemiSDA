
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.hooks import FixedThresholdingHook

from .utils import MocoBuildHook, refine_predictions, info_nce_loss, diversification_loss, update_labels


@ALGORITHMS.register('contratta')
class ContraTTA(AlgorithmBase):
    """
        GuidingPL algorithm (https://arxiv.org/abs/2303.03770).

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
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # flexmatch specified arguments
        self.init(queue_size=args.queue_size, m=args.m, T_moco=args.T_moco, refine_method=args.refine_method, \
                ce_sup_type=args.ce_sup_type, num_neighbors=args.num_neighbors, dist_type=args.dist_type, \
                contrast_type=args.contrast_type, alpha=args.alpha, beta=args.beta, eta=args.eta)
    
    def init(self, queue_size, m, T_moco, refine_method, ce_sup_type, num_neighbors, dist_type, contrast_type, alpha, beta, eta):
        self.queue_size = queue_size
        self.m = m
        self.T_moco = T_moco
        self.refine_method = refine_method
        self.ce_sup_type = ce_sup_type
        self.num_neighbors = num_neighbors
        self.dist_type = dist_type
        self.contrast_type = contrast_type
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

    def set_hooks(self):
        self.register_hook(MocoBuildHook(), "MocoBuildHook")
        super().set_hooks()

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, y_ulb, x_ulb_w, x_ulb_s_0, x_ulb_s_1):
        num_lb = y_lb.shape[0] 

        zero_tensor = torch.tensor([0.0]).to("cuda")
        with self.amp_cm():
            inputs = torch.cat((x_lb, x_ulb_w))
            feats_out, logits_out = self.moco_model(inputs, cls_only=True)
            logits_x_lb = logits_out[:num_lb]
            logits_w = logits_out[num_lb:]
            feats_x_lb = feats_out[:num_lb]
            feats_w = feats_out[num_lb:]

            with torch.no_grad():
                probs_w = F.softmax(logits_w, dim=1)
                pseudo_labels_w, probs_w, _ = refine_predictions(feats_w, probs_w, self.hooks_dict['MocoBuildHook'].banks, args=self.args)
            
            feats_q, logits_q, logits_ins, keys = self.moco_model(x_ulb_s_0, x_ulb_s_1)
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_w, 'x_ulb_s_0':feats_q, 'x_ulb_s_1':keys}

            if self.args.num_labels == 0:
                sup_loss = torch.tensor(0.).cuda(self.args.gpu)
            else:
                sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            self.moco_model.update_memory(keys, pseudo_labels_w)

            loss_nce = info_nce_loss(
                logits_ins=logits_ins,
                pseudo_labels=pseudo_labels_w,
                mem_labels=self.moco_model.mem_labels,
                contrast_type=self.contrast_type,)
            
            unsup_loss = self.consistency_loss(logits_q, pseudo_labels_w, 'ce', mask=None)
            
            loss_div = diversification_loss(logits_w, logits_q, self.args) if self.eta > 0 else zero_tensor

            total_loss = sup_loss + self.alpha * unsup_loss + self.beta * loss_nce + self.eta * loss_div

            with torch.no_grad():
                outs = self.moco_model.momentum_model(x_ulb_w)
                logits_w = outs['logits']
                feats_w = outs['feat']

            update_labels(self.hooks_dict['MocoBuildHook'].banks, idx_ulb, feats_w, logits_w, self.args)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         loss_nce=loss_nce.item(),
                                         unsup_loss=unsup_loss.item(), 
                                         loss_div=loss_div.item(), 
                                         total_loss=total_loss.item(), )
        return out_dict, log_dict
        

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['pseudo_item_list'] = self.hooks_dict['MocoBuildHook'].pseudo_item_list
        banks = dict()
        for key, value in self.hooks_dict['MocoBuildHook'].banks.items():
            if isinstance(value, torch.Tensor):
                banks[key] = value.cpu()
            else:
                banks[key] = value
        save_dict['banks'] = banks


        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        if 'banks' not in checkpoint:
            self.print_fn("additional parameter not found")
        else:
            self.hooks_dict['MocoBuildHook'].pseudo_item_list = checkpoint['pseudo_item_list']
            banks = checkpoint['banks']  
            for key, value in banks.items():
                if isinstance(value, torch.Tensor):
                    banks[key] = value.cuda(self.args.gpu)
            self.hooks_dict['MocoBuildHook'].banks = banks
            self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--queue_size', int, 16384),
            SSL_Argument('--m', float, 0.999),
            SSL_Argument('--T_moco', float, 0.07),
            SSL_Argument('--refine_method', str, "nearest_neighbors"),
            SSL_Argument('--ce_sup_type', str, "weak_strong"),
            SSL_Argument('--num_neighbors', int, 10),
            SSL_Argument('--dist_type', str, "cosine"),
            SSL_Argument('--contrast_type', str, "class_aware"),
            SSL_Argument('--alpha', float, 1.0),
            SSL_Argument('--beta', float, 1.0),
            SSL_Argument('--eta', float, 1.0),
        ]

