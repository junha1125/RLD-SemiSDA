# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function
import numpy as np
from sklearn.metrics import confusion_matrix
from copy import deepcopy
from collections import Counter
import tqdm

from semilearn.core.hooks import Hook
from semilearn.core.utils import send_model_cuda 


class MmeBuildHook(Hook):
    """
        Build MME archetecture.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def before_run(self, algorithm):
        if 'resnet' in algorithm.args.net:
            algorithm.model = MmeResNet(algorithm.model.module)
        elif 'vit' in algorithm.args.net:
            algorithm.model = MmeViT(algorithm.model.module)
        else:
            raise NotImplementedError("MME only works with resnet and vit models")
        algorithm.model = send_model_cuda(algorithm.args, algorithm.model, clip_batch=False)


class MmeResNet(nn.Module):
    def __init__(self, resnet_base):
        super().__init__()
        self.encoder = resnet_base.encoder
        self.fc = resnet_base.fc
        self.output_dim = resnet_base.output_dim

    def forward(self, x, only_fc=False, only_feat=False, reverse=False, **kwargs):
        if only_fc:
            if reverse:
                x = grad_reverse(x)
                return self.fc(x)
            else:
                return self.fc(x)
        elif only_feat:
            x = self.encoder(x)
            x = torch.flatten(x, 1)
            return x
        else:
            x = self.encoder(x)
            x = torch.flatten(x, 1)
            out = self.fc(x)
            result_dict = {'logits':out, 'feat':x}
            return result_dict


# TODO find a better way
class MmeViT(nn.Module):
    def __init__(self, vit_base):
        super().__init__()
        self.cls_token = vit_base.cls_token
        self.pos_embed = vit_base.pos_embed
        self.patch_embed = vit_base.patch_embed
        self.blocks = vit_base.blocks
        self.head = vit_base.head
        self.norm = vit_base.norm
        self.extract = vit_base.extract
        self.global_pool = vit_base.global_pool
        self.fc_norm = vit_base.fc_norm

    def forward(self, x, only_fc=False, only_feat=False, reverse=False, **kwargs):
        if only_fc:
            if reverse:
                x = grad_reverse(x)
                return self.head(x)
            else:
                return self.head(x)
        
        x = self.extract(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        
        if only_feat:
            return x
        
        output = self.head(x)
        result_dict = {'logits':output, 'feat':x}
        return result_dict
        

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambd
        return output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def advbce_unlabeled(target, f, prob, prob1):
    """Construct adversarial adpative clustering loss."""
    bce = BCE_softlabels()
    f = F.normalize(f) / 0.05
    target_ulb = pairwise_target(f, target)
    prob_bottleneck_row, _ = PairEnum2D(prob)
    _, prob_bottleneck_col = PairEnum2D(prob1)
    adv_bce_loss = -bce(prob_bottleneck_row, prob_bottleneck_col, target_ulb)
    return adv_bce_loss


def pairwise_target(f, target):
    """Produce pairwise similarity label."""
    fd = f.detach()
    # For unlabeled data
    if target is None:
        rank_feat = fd
        rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
        rank_idx1, rank_idx2 = PairEnum2D(rank_idx)
        rank_idx1, rank_idx2 = rank_idx1[:, :5], rank_idx2[:, :5]
        rank_idx1, _ = torch.sort(rank_idx1, dim=1)
        rank_idx2, _ = torch.sort(rank_idx2, dim=1)
        rank_diff = rank_idx1 - rank_idx2
        rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
        target_ulb = torch.ones_like(rank_diff).float().cuda()
        target_ulb[rank_diff > 0] = 0
    # For labeled data
    elif target is not None:
        raise NotImplementedError
    else:
        raise ValueError("Please check your target.")
    return target_ulb


def PairEnum1D(x):
    """Enumerate all pairs of feature in x with 1 dimension."""
    assert x.ndimension() == 1, "Input dimension must be 1"
    x1 = x.repeat(
        x.size(0),
    )
    x2 = x.repeat(x.size(0)).view(-1, x.size(0)).transpose(1, 0).reshape(-1)
    return x1, x2


def PairEnum2D(x):
    """Enumerate all pairs of feature in x with 2 dimensions."""
    assert x.ndimension() == 2, "Input dimension must be 2"
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    return x1, x2


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class BCE(nn.Module):
    eps = 1e-7

    def forward(self, prob1, prob2, simi):
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()
    

class BCE_softlabels(nn.Module):
    """Construct binary cross-entropy loss."""

    eps = 1e-7

    def forward(self, prob1, prob2, simi):
        P = prob1.mul_(prob2)
        P = P.sum(1)
        neglogP = -(
            simi * torch.log(P + BCE.eps) + (1.0 - simi) * torch.log(1.0 - P + BCE.eps)
        )
        return neglogP.mean()
