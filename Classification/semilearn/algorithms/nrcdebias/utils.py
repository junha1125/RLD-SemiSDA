# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from sklearn.metrics import confusion_matrix
from copy import deepcopy
from collections import Counter
import tqdm

from semilearn.algorithms.hooks import MaskingHook
from semilearn.core.hooks import Hook
from semilearn.core.utils import get_data_loader 

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class ObtainNRCLabel(Hook):
    """
    Obtain pseudo labels using NRC algorithm.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_bank = torch.tensor(0)
        self.fea_bank = torch.tensor(0)

    def before_run(self, algorithm):
        self.K = algorithm.K
        self.KK = algorithm.KK
        # build a replica of train_ulb dataset for pseudo label generation
        dataset_train_ulb = deepcopy(algorithm.dataset_dict['train_ulb'])
        dataset_train_ulb.is_ulb = False
        algorithm.loader_dict['train_ulb_replica'] = get_data_loader(algorithm.args, dataset_train_ulb,
                                                                algorithm.args.eval_batch_size,
                                                                data_sampler=None, # Important!
                                                                num_workers=algorithm.args.num_workers,
                                                                distributed=algorithm.distributed,
                                                                drop_last=False)
        self.data_loader_for_gen_pseudo = algorithm.loader_dict['train_ulb_replica']
    
    def before_train_step(self, algorithm):
        if algorithm.it == 0:
            # build feature and score bank
            self.fea_bank, self.score_bank = build_bank(self.data_loader_for_gen_pseudo, 
                                                            algorithm.ema_model, algorithm.args, algorithm.print_fn)
            
    @torch.no_grad()
    def obtain_neighbors(self, algorithm, tar_idx, features_test, softmax_out):
        """
        Obtrain the scores and features of neighbors of the target sample.
            K: number of neighbors
            M=KK: number of neighbors of neighbors
            Dim: feature dimension
            C: number of classes
            n: number of samples in the dataset

        Output:
            score_near_kk: batch x KM x C
            weight_kk: batch x KM
            score_near: batch x K x C
        """
        output_f_norm = F.normalize(features_test)
        output_f_ = output_f_norm.detach().clone()
        self.fea_bank[tar_idx] = output_f_.clone()
        self.score_bank[tar_idx] = softmax_out.detach().clone()

        distance = output_f_ @ self.fea_bank.T
        _, idx_near = torch.topk(distance,dim=-1,largest=True,k=self.K+1)
        idx_near = idx_near[:, 1:]  #batch x K
        score_near = self.score_bank[idx_near]    #batch x K x C

        fea_near = self.fea_bank[idx_near]  #batch x K x num_dim
        fea_bank_re = self.fea_bank.unsqueeze(0).expand(fea_near.shape[0],-1,-1) # batch x n x dim
        distance_ = torch.bmm(fea_near, fea_bank_re.permute(0,2,1))  # batch x K x n
        _,idx_near_near=torch.topk(distance_,dim=-1,largest=True,k=self.KK+1)  # M near neighbors for each of above K ones
        idx_near_near = idx_near_near[:,:,1:] # batch x K x M
        tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
        match = (idx_near_near == tar_idx_).sum(-1).float()  # batch x K (among neighbors of target, ones including target as neighbor.)
        weight = torch.where(match > 0., match, torch.ones_like(match).fill_(0.1))  # batch x K

        weight_kk = weight.unsqueeze(-1).expand(-1, -1, self.KK)  # batch x K x M
        weight_kk = weight_kk.fill_(0.1)

        score_near_kk = self.score_bank[idx_near_near]  # batch x K x M x C
        weight_kk = weight_kk.contiguous().view(weight_kk.shape[0], -1)  # batch x KM
        score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1, algorithm.num_classes)  # batch x KM x C

        score_self = self.score_bank[tar_idx]
        
        return score_near, weight, score_near_kk, weight_kk


@torch.no_grad()
def build_bank(dataloader, ema_model, args, print_fn):
    """
    Obtain pseudo labels for unlabeled data under the ODA (Open-set Domain Adaptation) setting.
    """
    logits, gt_labels, indices = [], [], []
    features = []
    ema_model.eval()
    for data in dataloader:
        imgs = data['x_lb']
        labels = data['y_lb']
        idxs = data['idx_lb']
        
        if isinstance(imgs, dict):
            imgs = {k: v.cuda(args.gpu) for k, v in imgs.items()}
        else:
            imgs = imgs.cuda(args.gpu)
        labels = labels.cuda(args.gpu)
        idxs = idxs.cuda(args.gpu)

        # (B, D) x (D, K) -> (B, K)
        output = ema_model(imgs, cls_only=True)
        feats = output['feat']
        logits_cls = output['logits']

        features.append(feats)
        logits.append(logits_cls)
        gt_labels.append(labels)
        indices.append(idxs)

    ema_model.train()
    features = torch.cat(features)
    logits = torch.cat(logits)
    gt_labels = torch.cat(gt_labels)
    indices = torch.cat(indices)

    assert len(logits) == len(dataloader.dataset)

    #  detach().clone().cpu() / .cuda(algorithm.args.gpu)
    output_norm=F.normalize(features) # self.fea_bank 
    outputs=nn.Softmax(-1)(logits) # self.score_bank 
    
    return output_norm, outputs
