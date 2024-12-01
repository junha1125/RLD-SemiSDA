# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from sklearn.metrics import confusion_matrix
from copy import deepcopy
from collections import Counter
import tqdm

from semilearn.algorithms.hooks import MaskingHook
from semilearn.algorithms.utils import concat_all_gather, remove_wrap_arounds
from semilearn.core.hooks import Hook
from semilearn.core.utils import get_data_loader

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class ObtainClusterLabel(Hook):
    """
    Obtain cluster labels
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mem_label = torch.tensor(0)

    def before_run(self, algorithm):
        # build a replica of train_ulb dataset for pseudo label generation
        dataset_train_ulb = deepcopy(algorithm.dataset_dict['train_ulb'])
        dataset_train_ulb.is_ulb = False
        sampler = DistributedSampler(dataset_train_ulb, shuffle=False) if (algorithm.distributed and algorithm.world_size > 1) else None
        algorithm.loader_dict['train_ulb_replica'] = get_data_loader(algorithm.args, dataset_train_ulb,
                                                                algorithm.args.eval_batch_size,
                                                                data_sampler=sampler, # Important!
                                                                num_workers=algorithm.args.num_workers,
                                                                distributed=algorithm.distributed,
                                                                drop_last=False,)
        self.data_loader_for_gen_pseudo = algorithm.loader_dict['train_ulb_replica']
    
    def before_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_iter_per_epoch) or algorithm.it == 0:
            # obtain pseudo labels
            algorithm.model.eval()
            algorithm.ema.apply_shadow()
            obtained_output = obtain_label(self.data_loader_for_gen_pseudo, algorithm.model, algorithm.args, algorithm.print_fn)
            self.mem_label = torch.from_numpy(obtained_output).cuda(algorithm.args.gpu)
            algorithm.ema.restore()
            algorithm.model.train()

    @torch.no_grad()
    def obtain_current_label(self, algorithm, tar_idx):
        return self.mem_label[tar_idx]


@torch.no_grad()
def obtain_label(dataloader, model, args, print_fn):
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
        output = model(imgs, cls_only=True)
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

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
    unknown_weight = 1 - ent / np.log(args.num_classes)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1).cuda(args.gpu)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    # kmeans clustering with known samples (hard coding)
    all_fea = all_fea.float().cpu().numpy() 
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()  

    # recalculate the centroid and pseudo label
    predict = predict.cpu()
    for round in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().cpu().numpy()) / len(all_fea)
    log_str = 'Pred accuracy = {:.2f}% ->, Pseudo Accuracy = {:.2f}%'.format(accuracy * 100, acc * 100)
    print_fn(log_str)
    
    return predict.astype('int')


@torch.no_grad()
def obtain_label_oda(dataloader, ema_model, args, print_fn):
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
    all_fea = features      # .float().cpu()
    all_output = logits     # .float().cpu()
    all_label = gt_labels   # .cpu()

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1).cuda(args.gpu)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1) / np.log(args.num_classes)
    ent = ent.float().cpu()

    # saparate the known and unknown by entropy. 
    kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
    labels = kmeans.predict(ent.reshape(-1,1)) 

    # identify whether the cluster with 1 is known(low entropy) or unknown(high entropy)
    idx = np.where(labels==1)[0]
    iidx = 0
    if ent[idx].mean() > ent.mean():
        iidx = 1
    known_idx = np.where(kmeans.labels_ != iidx)[0]

    # remove the unknown samples
    all_fea = all_fea[known_idx,:]
    all_output = all_output[known_idx,:]
    predict = predict[known_idx]
    all_label_idx = all_label[known_idx]
    # calculate the entropy threshold with two centroid entropy (mean(2x1))
    ENT_THRESHOLD = (kmeans.cluster_centers_).mean()

    # kmeans clustering with known samples (hard coding)
    # only consider the class which is predicted more than threshold
    all_fea = all_fea.float().cpu().numpy() 
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()  
    initc = aff.transpose().dot(all_fea)    # feature centroid: num class x feature dimension 
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])  
    cls_count = np.eye(K)[predict.cpu()].sum(axis=0)  # num predicted class 
    labelset = np.where(cls_count>args.threshold)    
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance) # distance between feature and centroid
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    # check the number of each class: np.eye(K)[pred_label].sum(axis=0)

    # recalculate the centroid and pseudo label
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    # unknown label is set to num_classes, which is not considered in the loss function
    guess_label = args.num_classes * np.ones(len(all_label), ) 
    # assign the pseudo label to the known samples
    guess_label[known_idx] = pred_label    

    acc = np.sum(guess_label == all_label.float().cpu().numpy()) / len(all_label_idx)
    log_str = 'Ent threshold = {:.2f}, Pred accuracy = {:.2f}% ->, Pseudo Accuracy (only samples with low entropy) {:.2f}%'.format(ENT_THRESHOLD, accuracy*100, acc*100)
    print_fn(log_str)
    
    return guess_label.astype('int'), ENT_THRESHOLD

def compute_entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 
        