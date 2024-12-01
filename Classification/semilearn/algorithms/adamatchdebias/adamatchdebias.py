# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


from .utils import AdaMatchThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook, DebiasSamplingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.hooks.debias_sampling import get_append_type


@ALGORITHMS.register('adamatchdebias')
class AdaMatchDebias(AlgorithmBase):
    """
        AdaMatch algorithm (https://arxiv.org/abs/2106.04732).

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
            - ema_p (`float`):
                momentum for average probability
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(p_cutoff=args.p_cutoff, T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, \
                  reliable_rate=args.reliable_rate, num_append=args.num_append,)
        self.append_type = get_append_type(args.append_type)
    
    def init(self, p_cutoff, T, hard_label=True, ema_p=0.999, reliable_rate=0.6, num_append=3):
        self.p_cutoff = p_cutoff
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.reliable_rate = reliable_rate
        self.num_append = num_append


    def set_hooks(self):
        self.register_hook(DebiasSamplingHook(), "DebiasSamplingHook")
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='model'), 
            "DistAlignHook")
        self.register_hook(AdaMatchThresholdingHook(), "MaskingHook")
        super().set_hooks()


    def train_step(self, idx_lb, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb_origin = y_lb.shape[0]
        x_lb, y_lb, y_lb_gt = self.call_hook(self.append_type, "DebiasSamplingHook", idx_lb, x_lb, y_lb, self.num_append)
        num_lb = y_lb.shape[0]
        num_lb_add = y_lb.shape[0] - num_lb_origin

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                print('just before inference, {}'.format(inputs.shape))
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
                ce_loss = self.ce_loss(logits_x_lb, y_lb, reduction='none')
                sup_loss = ce_loss[:num_lb_origin].mean() + ce_loss[num_lb_origin:].mean() * 1/self.num_append

            # probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
            probs_x_lb = self.compute_prob(logits_x_lb[:num_lb_origin].detach())
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb.clone())

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_lb=probs_x_lb, logits_x_ulb=probs_x_ulb_w, softmax_x_lb=False, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            # calculate loss
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
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        # DebiasSamplingHook
        save_dict['indices_memory'] = self.hooks_dict['DebiasSamplingHook'].indices.cpu()
        save_dict['preds_memory'] = self.hooks_dict['DebiasSamplingHook'].preds.cpu()
        save_dict['feats_memory'] = self.hooks_dict['DebiasSamplingHook'].feats.cpu()
        save_dict['logits_memory'] = self.hooks_dict['DebiasSamplingHook'].logits.cpu()
        save_dict['probs_memory'] = self.hooks_dict['DebiasSamplingHook'].probs.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        if 'p_model' not in checkpoint:
            self.print_fn("additional parameter not found")
        else:
            self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
            self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
            # DebiasSamplingHook
            self.hooks_dict['DebiasSamplingHook'].indices = checkpoint['indices_memory'].cuda(self.args.gpu)
            self.hooks_dict['DebiasSamplingHook'].preds = checkpoint['preds_memory'].cuda(self.args.gpu)
            self.hooks_dict['DebiasSamplingHook'].feats = checkpoint['feats_memory'].cuda(self.args.gpu)
            self.hooks_dict['DebiasSamplingHook'].logits = checkpoint['logits_memory'].cuda(self.args.gpu)
            self.hooks_dict['DebiasSamplingHook'].probs = checkpoint['probs_memory'].cuda(self.args.gpu)
            self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--reliable_rate', float, 0.4),
            SSL_Argument('--num_append', int, 3),
            SSL_Argument('--append_type', str, 'random'), # kmeans, random, opposite
        ]