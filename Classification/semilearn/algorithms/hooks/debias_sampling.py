# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from copy import deepcopy
import time

from semilearn.algorithms.utils import concat_all_gather, remove_wrap_arounds
from semilearn.core.hooks import Hook
from semilearn.core.utils import get_data_loader

from scipy.spatial.distance import cdist


def get_append_type(append_type):
    if append_type == 'none':
        return 'append_nothing'
    elif append_type == 'random':
        return 'append_random_samclass_samples'
    elif append_type == 'kmeans':
        return 'append_kmeans_samclass_samples'
    elif append_type == 'opposite':
        return 'append_opposite_samclass_samples'
    else:
        raise NotImplementedError(f"append_type {append_type} is not implemented")


class DebiasSamplingHook(Hook):
    """
    Incorrectly predicted samples cuase the biased model.
    This hook is to select the debiased samples.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.preds, self.feats = torch.tensor(0), torch.tensor(0)
        self.logits, self.probs = torch.tensor(0), torch.tensor(0)


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
        self.warming_up_epochs = 2
        self.gen_label_period = 2
        self.gen_label_iter = algorithm.num_iter_per_epoch // self.gen_label_period
        self.ds_train_ulb = deepcopy(algorithm.dataset_dict['train_ulb'])
        self.ds_train_ulb.is_ulb = False

        # get indices of unlabeled data among labeled data
        lb_data = algorithm.dataset_dict['train_lb'].data
        self.lb_index_in_ulb = torch.tensor([np.where(self.ds_train_ulb.data == sample)[0][0] for sample in lb_data], device=algorithm.gpu)
        assert len(self.lb_index_in_ulb) == len(lb_data)
    

    def before_train_step(self, algorithm):
        # Set the gen_label_iter according to the epoch
        if algorithm.epoch >= self.warming_up_epochs:
            self.gen_label_period = 1
            self.gen_label_iter = algorithm.num_iter_per_epoch // self.gen_label_period

        if self.every_n_iters(algorithm, self.gen_label_iter) or algorithm.it == 0:
            # Step1: obtain pseudo labels
            algorithm.model.eval()
            self.indices, self.preds, self.feats, self.logits, self.probs = self.obtain_memories(\
                                                    algorithm.it//self.gen_label_iter, algorithm.model, algorithm.args, algorithm.print_fn)
            algorithm.model.train()
            self.lb_pred = self.preds[self.lb_index_in_ulb].clone()
            self.lb_feats = self.feats[self.lb_index_in_ulb].clone()
            
            # Step2: remove unreliable samples
            self.indices, self.preds, self.feats, self.logits, self.probs, self.mask_to_keep = filter_reliable(\
                        self.indices, self.preds, self.feats, self.logits, self.probs, algorithm.args.num_classes, reliable_rate=algorithm.args.reliable_rate)
            
            # Step3: save in cache
            algorithm.print_fn(f"[MemoryLabelHook] Saving pseudo labels in cache..."); s_time = time.time()
            # self.data_cache = [self.ds_train_ulb.__getitem__(idx)['x_lb'] for idx in self.indices]
            dataset_dump = deepcopy(self.ds_train_ulb)
            dataset_dump.data = self.ds_train_ulb.data[self.indices.cpu()]
            dataset_dump.targets = self.preds.cpu()
            dataloader_dump = get_data_loader(algorithm.args, dataset_dump,
                                              algorithm.args.eval_batch_size,
                                              data_sampler=None, 
                                              num_workers=algorithm.args.num_workers,
                                              distributed=algorithm.distributed,
                                              drop_last=False,)
            self.data_cache = []
            for data in dataloader_dump:
                self.data_cache.extend(data['x_lb'])
            algorithm.print_fn(f"[MemoryLabelHook] Saving pseudo labels in cache... Done (time:{time.time() - s_time:.2f}s)")


    def append_nothing(self, algorithm, idx_lb, x_lb, y_lb, num_append=3):
        return x_lb, y_lb


    @torch.no_grad()
    def append_random_samclass_samples(self, algorithm, idx_lb, x_lb, y_lb, num_append=3):
        import time; start=time.time()
        add_x_lb = []
        add_y_lb = []
        # add_y_lb_gt = [] # after finishing debug, delete all code related to this variable
        for y in y_lb:
            idxs = (self.preds == y).nonzero(as_tuple=True)[0]
            idxs_selected = idxs[ torch.randperm(len(idxs))[:num_append] ]
            for i, idx in enumerate( self.indices[idxs_selected] ):  ## WARNING!!
                # add_x_lb.append(self.ds_train_ulb.__getitem__(idx)['x_lb'].cuda(algorithm.gpu))
                add_x_lb.append(self.data_cache[idxs_selected[i]].cuda(algorithm.gpu))
                add_y_lb.append(y.cuda(algorithm.gpu))
                # add_y_lb_gt.append(torch.tensor(self.ds_train_ulb.__getitem__(idx)['y_lb']))
        # add_y_lb_gt = torch.stack(add_y_lb_gt, dim=0).cuda(algorithm.gpu)
        # y_lb_gt = torch.cat((y_lb, add_y_lb_gt), dim=0)
        
        add_x_lb = torch.stack(add_x_lb, dim=0)
        add_y_lb = torch.stack(add_y_lb, dim=0)
        x_lb = torch.cat((x_lb, add_x_lb), dim=0)
        y_lb = torch.cat((y_lb, add_y_lb), dim=0)
        return x_lb, y_lb, None
    
    

    @torch.no_grad()
    def append_kmeans_samclass_samples(self, algorithm, idx_lb, x_lb, y_lb, num_append=3):
        from sklearn.cluster import KMeans
        add_x_lb = []
        add_y_lb = []
        add_y_lb_gt = [] # after finishing debug, delete all code related to this variable
        for y in y_lb:
            idxs = (self.preds == y).nonzero(as_tuple=True)[0]
            if len(idxs) < num_append+1:
                idxs_selected = torch.arange(len(idxs)).cuda(algorithm.gpu)
            else:
                kemans_results = KMeans(n_clusters=num_append).fit(self.feats[idxs].cpu().numpy()) 
                centers = kemans_results.cluster_centers_
                similarity = torch.tensor(centers).cuda(algorithm.gpu) @ self.feats[idxs].T 
                idxs_selected = torch.argmax(similarity, dim=1)
            indices_in_class = self.indices[idxs]
            for idx in indices_in_class[idxs_selected]:  ## WARNING!!
                add_x_lb.append(self.ds_train_ulb.__getitem__(idx)['x_lb'].cuda(algorithm.gpu))
                add_y_lb.append(y.cuda(algorithm.gpu))
                add_y_lb_gt.append(torch.tensor(self.ds_train_ulb.__getitem__(idx)['y_lb']))
        add_y_lb_gt = torch.stack(add_y_lb_gt, dim=0).cuda(algorithm.gpu)
        y_lb_gt = torch.cat((y_lb, add_y_lb_gt), dim=0)
        
        add_x_lb = torch.stack(add_x_lb, dim=0)
        add_y_lb = torch.stack(add_y_lb, dim=0)
        x_lb = torch.cat((x_lb, add_x_lb), dim=0)
        y_lb = torch.cat((y_lb, add_y_lb), dim=0)
        return x_lb, y_lb, y_lb_gt
        # if algorithm.rank == 0 or torch.any(idxs_selected >= len(indices_in_class)): print(">>> {} |||| {}".format(len(indices_in_class), idxs_selected)) 
        # if algorithm.rank == 0 or torch.any(idxs_selected >= len(indices_in_class)): print(">>> {} \n".format(idxs_selected >= len(indices_in_class))) 
    

    def append_opposite_samclass_samples(self, algorithm, idx_lb, x_lb, y_lb, num_append=3):
        add_x_lb = []
        add_y_lb = []
        add_y_lb_gt = [] # after finishing debug, delete all code related to this variable
        for i_lb, y in zip(idx_lb, y_lb):
            idxs = (self.preds == y).nonzero(as_tuple=True)[0]
            if len(idxs) < num_append+1:
                idxs_selected = torch.arange(len(idxs)).cuda(algorithm.gpu)
            else:
                similarity = self.lb_feats[i_lb] @ self.feats[idxs].T
                idxs_selected = torch.argsort(similarity, descending=False)[:num_append]
            indices_in_class = self.indices[idxs]
            for idx in indices_in_class[idxs_selected]:  ## WARNING!!
                add_x_lb.append(self.ds_train_ulb.__getitem__(idx)['x_lb'].cuda(algorithm.gpu))
                add_y_lb.append(y.cuda(algorithm.gpu))
                add_y_lb_gt.append(torch.tensor(self.ds_train_ulb.__getitem__(idx)['y_lb']))
        add_y_lb_gt = torch.stack(add_y_lb_gt, dim=0).cuda(algorithm.gpu)
        y_lb_gt = torch.cat((y_lb, add_y_lb_gt), dim=0)
        
        add_x_lb = torch.stack(add_x_lb, dim=0)
        add_y_lb = torch.stack(add_y_lb, dim=0)
        x_lb = torch.cat((x_lb, add_x_lb), dim=0)
        y_lb = torch.cat((y_lb, add_y_lb), dim=0)
        return x_lb, y_lb, y_lb_gt
    

    @torch.no_grad()
    def obtain_memories(self, curr_hist_idx, model, args, print_fn):
        print_fn('[MemoryLabelHook] Generating pseudo labels...'); s_time = time.time()
        dataloader = self.data_loader_for_gen_pseudo
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
        features = F.normalize(features)
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

        gt_labels = gt_labels   # .cpu() / according to your gpu memory, you can change it to .cpu() # (N)
        indices = indices       # .cpu() # (N)
        feats = features        # .float().cpu() / F.normalize(all_fea) # (N x d)
        logits = logits         # .float().cpu() # (N x C)
        probs = nn.Softmax(dim=1)(logits)    # (N x C)
        max_probs, pred_labels = torch.max(probs, 1)  # (N), (N)
        print_fn('[MemoryLabelHook] Generating pseudo labels... Done (time:{:.2f}s)'.format(time.time() - s_time))
        return indices, pred_labels, feats, logits, probs

        # Filter out unreliable samples
        # indeces, pred_labels, feats, logits, probs, mask_to_keep = \
        #     filter_reliable(args.num_classes, indices, pred_labels, feats, logits, probs, reliable_rate=args.reliable_rate)
        # cls_count = np.eye(args.num_classes)[pred_labels.cpu()].sum(axis=0)
        # acc_pred = torch.sum(torch.squeeze(pred_labels).float() == gt_labels[mask_to_keep]).item() / len(feats)
        # return indeces, pred_labels, feats, logits, probs, mask_to_keep
    

def filter_reliable(indices, pred_labels, feats, logits, probs, num_classes, reliable_rate=0.6):
    max_probs, _ = torch.max(probs, 1)  # (N), (N)
    indices_samples_to_keep = {}

    for class_label in range(num_classes):
        class_indices = (pred_labels == class_label).nonzero(as_tuple=True)[0]
        sorted_indices = torch.argsort(max_probs[class_indices], descending=True)
        num_samples_class = len(class_indices)
        num_samples_to_keep_class = int(reliable_rate * num_samples_class)
        indices_to_keep_class = class_indices[sorted_indices[:num_samples_to_keep_class]]
        indices_samples_to_keep[class_label] = indices_to_keep_class

    mask_to_keep = torch.zeros(len(max_probs), dtype=torch.bool)
    for class_label in range(num_classes):
        mask_to_keep[indices_samples_to_keep[class_label]] = True

    filtered_indices = indices[mask_to_keep]
    filtered_feats = feats[mask_to_keep]
    filtered_logits = logits[mask_to_keep]
    filtered_probs = probs[mask_to_keep]
    filtered_max_probs = max_probs[mask_to_keep]
    filtered_pred_labels = pred_labels[mask_to_keep]
    return filtered_indices, filtered_pred_labels, filtered_feats, filtered_logits, filtered_probs, mask_to_keep
    


def similarity_classifier(all_feats, aff, predict, num_classes):
    """
        aff: probabilities
    """
    for round in range(2):
        initc = aff.transpose().dot(all_feats)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(num_classes)[predict].sum(axis=0)
        labelset = np.where(cls_count>0.0)
        labelset = labelset[0]

        dd = cdist(all_feats, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(num_classes)[predict]
    return predict

