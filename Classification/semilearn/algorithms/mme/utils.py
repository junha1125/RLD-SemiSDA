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