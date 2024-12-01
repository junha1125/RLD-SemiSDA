# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook


class FreeMatchThresholingHook(MaskingHook):
    """
    SAT in FreeMatch
    """
    def __init__(self, num_classes, momentum=0.999, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.m = momentum
        
        self.p_model = torch.ones((self.num_classes)) / self.num_classes
        self.label_hist = torch.ones((self.num_classes)) / self.num_classes
        self.time_p = self.p_model.mean()
    
    @torch.no_grad()
    def update(self, algorithm, probs_x_ulb):
        if algorithm.distributed and algorithm.world_size > 1:
            probs_x_ulb = concat_all_gather(probs_x_ulb)
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1,keepdim=True)

        if algorithm.use_quantile:
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(max_probs,0.8) #* max_probs.mean()
        else:
            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
        
        if algorithm.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype) 
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())

        algorithm.p_model = self.p_model 
        algorithm.label_hist = self.label_hist 
        algorithm.time_p = self.time_p 
    

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(algorithm, probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        mask = max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)
        return mask


        


import pandas as pd
import os

import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from semilearn.core.hooks import Hook
from semilearn.core.utils import get_data_loader 
from scipy.spatial.distance import cdist


class HistorySaveHook(Hook):
    """
    Save the prediction history of the model as a numpy file.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mem_label = torch.tensor(0)

    def before_run(self, algorithm):
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
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
            self.checking_iter = algorithm.num_iter_per_epoch // 2
            
            algorithm.args.norm_method = 'nrc'
            self.history_path = os.path.join(algorithm.save_dir, algorithm.save_name, 'history.npy')
            if os.path.isfile(self.history_path):
                self.history = np.load(self.history_path, allow_pickle=True)
            else:
                labeled_masks = np.isin(algorithm.dataset_dict['train_ulb'].data, algorithm.dataset_dict['train_lb'].data)
                data_names = [os.path.join(str(data).split('/')[-2], str(data).split('/')[-1]) for data in dataset_train_ulb.data]
                self.history = [dict({'name': data_names, 'is_labeled': labeled_masks, 'gt_label': dataset_train_ulb.targets}) for _ in range(algorithm.epochs * 2 + 1)] 
            
    
    def before_train_step(self, algorithm):
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            if self.every_n_iters(algorithm, self.checking_iter) or algorithm.it == 0:
                # obtain pseudo labels
                algorithm.model.eval()
                algorithm.ema.apply_shadow()
                history_idx = algorithm.it // self.checking_iter
                save_history(self.data_loader_for_gen_pseudo, self.history, history_idx, algorithm.model, algorithm.args, algorithm.print_fn)
                algorithm.ema.restore()
                algorithm.model.train()
                np.save(self.history_path, self.history)


@torch.no_grad()
def save_history(dataloader, history, history_idx, model, args, print_fn):
    """
    Save the prediction history of the model as a numpy file.
    """
    logits, gt_labels, indices, preds = [], [], [], []
    features = []
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
        output = model(imgs, cls_only=True)
        feats = output['feat']
        logits_cls = output['logits']
        pred = torch.argmax(logits_cls, dim=1)

        features.append(feats)
        logits.append(logits_cls)
        gt_labels.append(labels)
        indices.append(idxs)
        preds.append(pred)

    features = torch.cat(features).cpu()
    logits = torch.cat(logits)
    probs = F.softmax(logits, dim=1).cpu().numpy()
    logits = logits.cpu().numpy()
    gt_labels = torch.cat(gt_labels).cpu().numpy()
    indices = torch.cat(indices).cpu().numpy()
    preds = torch.cat(preds).cpu().numpy()
    
    # find neerest neighbors
    if args.norm_method == 'nrc':
        features_norm=F.normalize(features)
    elif args.norm_method == 'shot':
        features = torch.cat((features, torch.ones(features.size(0), 1)), 1)
        features_norm = (features.t() / torch.norm(features, p=2, dim=1)).t()
    else:
        raise NotImplementedError
    features = features.numpy()
    features_norm = features_norm.cpu()
    distance = features_norm @ features_norm.T
    _, idx_near = torch.topk(distance, dim=-1, largest=True, k=5+1)
    idx_near = idx_near[:, 1:].numpy()  # Num_sample x K

    # similarity-based label using centroids
    aff = probs
    all_fea = features_norm.numpy()
    K = aff.shape[1]
    preds_sim = np.copy(preds)

    for round in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[preds_sim].sum(axis=0)
        # labelset = np.where(cls_count>0)[0]

        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        preds_sim = pred_label

        aff = np.eye(K)[preds_sim]

    accuracy = np.sum(preds == gt_labels) / len(all_fea)
    acc = np.sum(preds_sim == gt_labels) / len(all_fea)
    log_str = 'Pred accuracy = {:.2f}% ->, Pseudo Accuracy = {:.2f}%'.format(accuracy * 100, acc * 100)
    print_fn(log_str)

    # save history
    history[history_idx]['pred_label'] = preds.astype(np.int32)
    history[history_idx]['sim_label'] = preds_sim.astype(np.int32)
    history[history_idx]['logits'] = logits
    history[history_idx]['logits_sim'] = dd
    history[history_idx]['feat'] = features
    history[history_idx]['near_idx'] = idx_near.astype(np.int32)
    history[history_idx]['near_pred_label'] = preds[idx_near].astype(np.int32)
    history[history_idx]['near_sim_label'] = preds_sim[idx_near].astype(np.int32)