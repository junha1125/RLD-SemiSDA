# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from copy import deepcopy

from semilearn.algorithms.utils import concat_all_gather, remove_wrap_arounds
from semilearn.core.hooks import Hook
from semilearn.core.utils import get_data_loader

from scipy.spatial.distance import cdist

class MemoryLabelHook(Hook):
    """
    Obtain cluster labels
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mem_label = torch.tensor(0)

    def before_run(self, algorithm):
        # build a replica of train_ulb dataset for pseudo label generation
        ds_train_ulb_replica = deepcopy(algorithm.dataset_dict['train_ulb'])
        ds_train_ulb_replica.is_ulb = False
        ds_train_ulb_replica.transform = algorithm.dataset_dict['eval'].transform
        sampler = DistributedSampler(ds_train_ulb_replica, shuffle=False) if (algorithm.distributed and algorithm.world_size > 1) else None
        algorithm.loader_dict['train_ulb_replica'] = get_data_loader(algorithm.args, ds_train_ulb_replica,
                                                                algorithm.args.eval_batch_size,
                                                                data_sampler=sampler, 
                                                                num_workers=algorithm.args.num_workers,
                                                                distributed=algorithm.distributed,
                                                                drop_last=False,)
        self.data_loader_for_gen_pseudo = algorithm.loader_dict['train_ulb_replica']
        self.checking_iter = algorithm.num_iter_per_epoch // 2
    
    def before_train_step(self, algorithm):
        if self.every_n_iters(algorithm, self.checking_iter) or algorithm.it == 0:
            # obtain pseudo labels
            algorithm.model.eval()
            # algorithm.ema.apply_shadow()
            curr_hist_idx = algorithm.it // self.checking_iter
            obtained_output = obtain_label(self.data_loader_for_gen_pseudo, curr_hist_idx, algorithm.model, algorithm.args, algorithm.print_fn)
            if curr_hist_idx == 0:
                self.mem_label = torch.from_numpy(obtained_output).cuda(algorithm.args.gpu).unsqueeze(0)
            else:
                self.mem_label = torch.cat((self.mem_label, torch.from_numpy(obtained_output).cuda(algorithm.args.gpu).unsqueeze(0)), dim=0)
            # algorithm.ema.restore()
            algorithm.model.train()

    @torch.no_grad()
    def obtain_last_label(self, algorithm, tar_idx):
        return self.mem_label[-1, tar_idx]
    
    @torch.no_grad()
    def obtain_history_label(self, algorithm, tar_idx):
        return self.mem_label[:, tar_idx]


@torch.no_grad()
def obtain_label(dataloader, curr_hist_idx, model, args, print_fn):
    """
    Obtain pseudo labels for unlabeled data under the ODA (Open-set Domain Adaptation) setting.
    """
    logits, gt_labels, indices = [], [], []
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
        output = model(imgs)
        feats = output['feat']
        logits_cls = output['logits']

        features.append(feats)
        logits.append(logits_cls)
        gt_labels.append(labels)
        indices.append(idxs)

    features = torch.cat(features)
    logits = torch.cat(logits)
    gt_labels = torch.cat(gt_labels)
    indices = torch.cat(indices)

    if args.distributed and args.world_size > 1:
        features = concat_all_gather(features)
        logits = concat_all_gather(logits)
        gt_labels = concat_all_gather(gt_labels)
        indices = concat_all_gather(indices)

        ranks = len(dataloader.dataset) % dist.get_world_size()
        features = remove_wrap_arounds(features, ranks)
        logits = remove_wrap_arounds(logits, ranks)
        gt_labels = remove_wrap_arounds(gt_labels, ranks)
        indices = remove_wrap_arounds(indices, ranks)

        sorted_indices = torch.argsort(indices)
        features = features[sorted_indices]
        logits = logits[sorted_indices]
        gt_labels = gt_labels[sorted_indices]
        indices = indices[sorted_indices]

    assert len(indices) == len(dataloader.dataset) and \
           torch.all(indices == torch.arange(len(dataloader.dataset)).cuda(args.gpu))

    all_fea = features      # .float().cpu()
    all_output = logits     # .float().cpu()
    all_label = gt_labels   # .cpu()

    ## 1. predcion accuracy
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    acc_pred = torch.sum(torch.squeeze(predict).float() == all_label).item() / len(all_fea)
    
    ## Normalize
    all_fea = F.normalize(all_fea)
    
    ## 2. sim_predction accuracy
    all_fea_sim = all_fea.float().cpu().numpy() 
    aff = all_output.float().cpu().numpy()  
    predict_sim = predict.cpu()
    predict_sim = similarity_classifier(all_fea_sim, aff, predict_sim, args.num_classes)
    acc_sim_pred = np.sum(predict_sim == all_label.float().cpu().numpy()) / len(all_fea_sim)
    
    # ## 3. prediction (domain removed) accuracy
    # ## domain removing
    # all_fea = svd_based_domain_remover(all_fea, num_basis=1).cuda(args.gpu) # N x d

    # fc_batch_size = 256
    # re_preds = []
    # for iter in range(len(all_fea)//fc_batch_size):
    #     if iter == (len(all_fea) // fc_batch_size) - 1:
    #         feature_maps = all_fea[iter * fc_batch_size:]
    #     else:
    #         feature_maps = all_fea[iter * fc_batch_size:(iter + 1) * fc_batch_size]
    #     re_logits_cls = model(feature_maps, only_fc=True)
    #     _, repred = torch.max(re_logits_cls, 1)
    #     re_preds.append(repred)
    # re_preds = torch.cat(re_preds)
    # assert len(re_preds) == len(all_fea)
    # acc_domain_removed_pred = torch.sum(torch.squeeze(re_preds).float() == all_label).item() / len(all_fea)

    # ## 4. sim_predction (domain removed) accuracy
    # all_fea_sim_2 = all_fea.float().cpu().numpy() 
    # aff_2 = all_output.float().cpu().numpy()  
    # predict_sim_2 = re_preds.cpu()
    # predict_sim_2 = similarity_classifier(all_fea_sim_2, aff_2, predict_sim_2, args.num_classes)
    # acc_domain_removed_sim_pred = np.sum(predict_sim_2 == all_label.float().cpu().numpy()) / len(all_fea_sim_2)

    # log_srt = '[MemoryLabelHook] H{}| pred_acc = {:.1f}%, sim_pred_acc = {:.1f}%, | dr_pred_acc = {:.1f}%, dr_sim_pred_acc = {:.1f}%'\
    #             .format(curr_hist_idx, acc_pred * 100, acc_sim_pred * 100, acc_domain_removed_pred * 100, acc_domain_removed_sim_pred * 100)
    
    log_srt = '[MemoryLabelHook] H{}| pred_acc = {:.1f}%, sim_pred_acc = {:.1f}%'.format(curr_hist_idx, acc_pred * 100, acc_sim_pred * 100, )
    print_fn(log_srt)
    
    if args.gen_pseudo_type == "sim_pred":
        return predict_sim.astype('int')
    elif args.gen_pseudo_type == "dr_sim_pred":
        return predict_sim_2.astype('int')
    elif args.gen_pseudo_type == "pred":
        return predict.cpu().numpy().astype('int')
    elif args.gen_pseudo_type == "dr_pred":
        return re_preds.cpu().numpy().astype('int')
    else:
        raise Exception("Unknown pseudo label type: {}".format(args.gen_pseudo_type))
    
def svd_based_domain_remover(feat, num_basis):
    U, S, V = torch.svd(feat.T)

    U = U[:, :num_basis]
    projection_matrix = U @ U.T

    scale = (feat ** 2).sum(dim=-1, keepdim=True).sqrt() # Nx1
    feat = feat - feat @ projection_matrix
    scale_hat = (feat ** 2).sum(dim=-1, keepdim=True).sqrt()
    feat = feat * (scale / scale_hat)
    return feat


def similarity_classifier(all_fea, aff, predict, num_classes):
    for round in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(num_classes)[predict].sum(axis=0)
        labelset = np.where(cls_count>0.0)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(num_classes)[predict]
    return predict