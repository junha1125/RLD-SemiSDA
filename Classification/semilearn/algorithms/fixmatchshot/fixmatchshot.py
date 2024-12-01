# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, MemoryLabelHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('fixmatchshot')
class FixMatchShot(AlgorithmBase):

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
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, gen_pseudo_type=args.gen_pseudo_type, label_smoothing=args.label_smoothing)
    
    def init(self, T, p_cutoff, hard_label=True, gen_pseudo_type="sim_pred", label_smoothing=0.0):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.gen_pseudo_type = gen_pseudo_type
        self.label_smoothing = label_smoothing
        if self.label_smoothing:
            self.p_cutoff = self.p_cutoff * (1-self.label_smoothing)
    
    def set_hooks(self):
        self.register_hook(MemoryLabelHook(), "MemoryLabelHook")
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s_0, x_ulb_s_1):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
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
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            if self.args.num_labels == 0:
                sup_loss = torch.tensor(0.).cuda(self.args.gpu)
            else:
                sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            if self.label_smoothing > 0:
                pseudo_label = self.call_hook("gen_ulb_targets_based_history", "PseudoLabelingHook", 
                                            logits=logits_x_ulb_w,
                                            history_label=self.call_hook("obtain_history_label","MemoryLabelHook", tar_idx=idx_ulb).T,
                                            label_smoothing=self.label_smoothing)
            else:
                pseudo_label = self.call_hook("obtain_last_label","MemoryLabelHook", tar_idx=idx_ulb)

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

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['mem_label'] = self.hooks_dict['MemoryLabelHook'].mem_label.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        if 'mem_label' not in checkpoint:
            self.print_fn("no additional parameter loaded")
        else:
            self.hooks_dict['MemoryLabelHook'].mem_label = checkpoint['mem_label'].cuda(self.args.gpu)
            self.print_fn("additional parameter loaded")
        return checkpoint        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--gen_pseudo_type', str, "pred"),
            SSL_Argument('--label_smoothing', float, 0.1),
        ]
