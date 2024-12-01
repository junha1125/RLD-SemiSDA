import torch
import torch.nn.functional as F

from .utils import FreeMatchThresholingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DebiasSamplingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.hooks.debias_sampling import get_append_type

# TODO: move these to .utils or algorithms.utils.loss
def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


@ALGORITHMS.register('freematchdebias')
class FreeMatchDeBias(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, use_quantile=args.use_quantile, \
                  clip_thresh=args.clip_thresh, reliable_rate=args.reliable_rate, num_append=args.num_append,)
        self.lambda_e = args.ent_loss_ratio
        self.append_type = get_append_type(args.append_type)


    def init(self, T, hard_label=True, ema_p=0.999, use_quantile=True, clip_thresh=False, reliable_rate=0.6, num_append=3):
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh
        self.reliable_rate = reliable_rate
        self.num_append = num_append


    def set_hooks(self):
        self.register_hook(DebiasSamplingHook(), "DebiasSamplingHook")
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FreeMatchThresholingHook(num_classes=self.num_classes, momentum=self.args.ema_p), "MaskingHook")
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
                # sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

                ce_loss = self.ce_loss(logits_x_lb, y_lb, reduction='none')
                sup_loss = ce_loss[:num_lb_origin].mean() + ce_loss[num_lb_origin:].mean() * 1/self.num_append
                # _, pred_labels = torch.max(logits_x_lb, 1); print(pred_labels == y_lb); print(pred_labels == y_lb_gt)

            # calculate mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)


            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)
            
            # calculate unlabeled loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)
            
            # calculate entropy loss
            if mask.sum() > 0:
               ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
            else:
               ent_loss = 0.0
            # ent_loss = 0.0
            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_e * ent_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['MaskingHook'].p_model.cpu()
        save_dict['time_p'] = self.hooks_dict['MaskingHook'].time_p.cpu()
        save_dict['label_hist'] = self.hooks_dict['MaskingHook'].label_hist.cpu()
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
            self.print_fn("no additional parameter loaded")
        else:
            self.hooks_dict['MaskingHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
            self.hooks_dict['MaskingHook'].time_p = checkpoint['time_p'].cuda(self.args.gpu)
            self.hooks_dict['MaskingHook'].label_hist = checkpoint['label_hist'].cuda(self.args.gpu)
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
            SSL_Argument('--ent_loss_ratio', float, 0.001),
            SSL_Argument('--use_quantile', str2bool, False),
            SSL_Argument('--clip_thresh', str2bool, False),
            SSL_Argument('--reliable_rate', float, 0.4),
            SSL_Argument('--num_append', int, 3),
            SSL_Argument('--append_type', str, 'random'), # kmeans, random, opposite
        ]
