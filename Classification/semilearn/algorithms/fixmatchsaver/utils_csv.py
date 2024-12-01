# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import os

import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from semilearn.core.hooks import Hook
from semilearn.core.utils import get_data_loader 


class HistorySaveHook(Hook):
    """
    Save the prediction history of the model as a **CSV** file.
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
            self.history = np.zeros((len(dataset_train_ulb) * 4, algorithm.num_train_iter//self.checking_iter+1)).astype(int)
            self.history_path = os.path.join(algorithm.save_dir, algorithm.save_name, 'history.csv')
            self.df_history = pd.DataFrame(self.history)
    
    def before_train_step(self, algorithm):
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            if self.every_n_iters(algorithm, self.checking_iter) or algorithm.it == 0:
                # obtain pseudo labels
                algorithm.model.eval()
                algorithm.ema.apply_shadow()
                history_idx = algorithm.it // self.checking_iter
                save_history(self.data_loader_for_gen_pseudo, self.df_history, history_idx, algorithm.model, algorithm.args, algorithm.print_fn)
                algorithm.ema.restore()
                algorithm.model.train()
                
                # hostory array to pandas dataframe
                if algorithm.it == 0:
                    data_names = get_data_names(self.data_loader_for_gen_pseudo, algorithm)
                    self.df_history.insert(loc=0, column='name', value=data_names)
                    index_names = get_indexs(self.data_loader_for_gen_pseudo)
                    self.df_history.insert(loc=0, column='index', value=index_names)
                self.df_history.to_csv(self.history_path, index=False)


def get_data_names(dataloader, algorithm):
    """
    Get the names of the data in the dataloader.
    """
    labeled_masks = np.isin(algorithm.dataset_dict['train_ulb'].data, algorithm.dataset_dict['train_lb'].data)
    data_names = []
    for data, mask in zip(dataloader.dataset.data, labeled_masks):
        data_name = os.path.join(str(data).split('/')[-2], str(data).split('/')[-1])
        data_names.append(data_name)
        second_line = '' if not mask else 'Labeled'
        data_names.append(second_line)
        data_names.append('')
        data_names.append('')
    return data_names


def get_indexs(dataloader):
    """
    Get the indices of the data in the dataloader.
    """
    indexs = []
    for i in range(len(dataloader.dataset)):
        indexs.append('idx '+str(i))
        indexs.append('')
        indexs.append('')
        indexs.append('')
    return indexs


@torch.no_grad()
def save_history(dataloader, history, history_idx, model, args, print_fn):
    """
    Obtain pseudo labels for unlabeled data under the ODA (Open-set Domain Adaptation) setting.
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

    features = torch.cat(features)
    logits = torch.cat(logits)
    gt_labels = torch.cat(gt_labels).cpu().numpy()
    indices = torch.cat(indices).cpu().numpy()
    preds = torch.cat(preds).cpu().numpy()
    
    features_norm=F.normalize(features).cpu()
    distance = features_norm @ features_norm.T
    _, idx_near = torch.topk(distance, dim=-1, largest=True, k=5+1)
    idx_near = idx_near[:, 1:]  # Num_sample x K

    for i in indices:
        history[history_idx][i*4]   = gt_labels[i]
        history[history_idx][i*4+1] = preds[i]
        history[history_idx][i*4+2] = str(True) if gt_labels[i] == preds[i] else str(False)
        history[history_idx][i*4+3] = ' ,'.join([str(i) for i in idx_near[i, :].cpu().tolist()])