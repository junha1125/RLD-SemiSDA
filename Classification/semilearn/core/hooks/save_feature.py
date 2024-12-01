# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from .hook import Hook

import torch
import numpy as np
from tqdm import tqdm

from copy import deepcopy
from semilearn.core.utils import get_data_loader

from sklearn.cluster import KMeans
import pickle

class SaveFeatureHook(Hook):
    """
        Save features of source prototypes and target features
    """

    def before_run(self, algorithm):
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            # get source prototypes
            self.source_prototypes, self.source_prototypes_y, _, _  = get_features(algorithm, \
                                            dataset='eval_src', num_features=100, select_mode='kmeans')
            # get target prototypes
            target_features, target_y, target_false_mask, target_labeled_mask = get_features(algorithm, \
                                            dataset='train_ulb', num_features=500, select_mode='random')
            save_features(algorithm, 0, self.source_prototypes, self.source_prototypes_y, \
                                        target_features, target_y, target_false_mask, target_labeled_mask)

    def after_train_epoch(self, algorithm):
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            # get target prototypes
            target_features, target_y, target_false_mask, target_labeled_mask = get_features(algorithm, \
                                            dataset='train_ulb', num_features=500, select_mode='random')
            save_features(algorithm, algorithm.epoch, self.source_prototypes, self.source_prototypes_y, \
                                        target_features, target_y, target_false_mask, target_labeled_mask)
    

def extract_feature(dataset, model, ema, gpu, args):
    """
        extract features, y_gts, and position of incorrectly predicted samples
    """
    # make loader
    temp_dataset = deepcopy(dataset)
    temp_dataset.is_ulb = False
    temp_loader =  get_data_loader(args, temp_dataset, args.eval_batch_size, 
                                    data_sampler=None, num_workers=args.num_workers, distributed=args.distributed, drop_last=False)
    # divide false and true samples
    model.eval()
    # ema.apply_shadow()
    y_gt = []
    y_pred = []
    index = []
    feature = []
    with torch.no_grad():
        for data in tqdm(temp_loader):
            x = data['x_lb']
            y = data['y_lb']
            idx = data['idx_lb']
            
            if isinstance(x, dict):
                x = {k: v.cuda(gpu) for k, v in x.items()}
            else:
                x = x.cuda(gpu)
            y = y.cuda(gpu)

            num_batch = y.shape[0]
            logits = model(x)['logits']
            features = model(x)['feat']
            feature.append(features.cpu())
            y_gt.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            index.extend(idx.tolist())
    # ema.restore()
    model.train()

    y_gt = np.array(y_gt)
    y_pred = np.array(y_pred)
    index = np.array(index)
    feature = torch.cat(feature, dim=0).numpy()
    
    false_mask = y_pred != y_gt
    false_mask = np.array(false_mask)

    return feature, y_gt, false_mask


def get_features(algorithm, dataset='eval_src', num_features=100, select_mode='random'):
    # get target prototypes
    if dataset == 'train_ulb': assert select_mode == 'random', "Only support random selection for train_ulb"
    features, y_gt, false_mask = extract_feature(algorithm.dataset_dict[dataset], \
                                    algorithm.model, algorithm.ema, algorithm.gpu, algorithm.args)
    algorithm.print_fn("[SaveFeatureHook] accuracy of {}: {:.1f} | data size : {}".format(dataset, 100*(1 - false_mask.sum()/false_mask.shape[0]), false_mask.shape[0]))
    # get labeled mask
    labeled_mask = np.zeros(features.shape[0], dtype=bool)
    if dataset == 'train_ulb':   
        labeled_mask = np.isin(algorithm.dataset_dict['train_ulb'].data, algorithm.dataset_dict['train_lb'].data)
    
    # save max num_features features for each class.
    features_box = []
    false_mask_box = []
    labeled_mask_box = []  
    y_box = []
    for c in range(algorithm.num_classes):
        class_indexs = y_gt == c
        features_c = features[class_indexs]
        false_mask_c = false_mask[class_indexs]
        labeled_mask_c = labeled_mask[class_indexs]

        dedug_str = "Class: {} Acc:{:.1f} #Samples:{} #Labeles:{}".format(c, 100*(1 - false_mask_c.sum()/false_mask_c.shape[0]), \
                                                                false_mask_c.shape[0], labeled_mask_c.sum())
        algorithm.print_fn("[SaveFeatureHook]" + dedug_str)
        
        if features_c.shape[0] > num_features:
            if features_c.shape[0] < num_features * 3: select_mode = 'random'
            if select_mode == 'kmeans':
                kmeans = KMeans(n_clusters=num_features, random_state=0).fit(features_c)
                features_c = kmeans.cluster_centers_
                false_mask_c = np.zeros(features_c.shape[0], dtype=bool)
                labeled_mask_c = np.zeros(features_c.shape[0], dtype=bool)
            elif select_mode == 'random':
                # Random select features
                idx = np.random.choice(features_c.shape[0], num_features, replace=False)
                
                # Add labeled samples
                labeled_idx = np.where(labeled_mask_c)[0]
                already_exist = np.isin(labeled_idx, idx)
                labeled_idx = labeled_idx[already_exist == False]
                idx = np.append(idx, labeled_idx)

                features_c = features_c[idx]
                false_mask_c = false_mask_c[idx]
                labeled_mask_c = labeled_mask_c[idx]
            else:
                raise NotImplementedError
        
        features_box.append(features_c)
        false_mask_box.append(false_mask_c)
        labeled_mask_box.append(labeled_mask_c)
        y_box.append(np.ones(features_c.shape[0]) * c)
    
    features_box = np.concatenate(features_box, axis=0)
    false_mask_box = np.concatenate(false_mask_box, axis=0)
    labeled_mask_box = np.concatenate(labeled_mask_box, axis=0)
    y_box = np.concatenate(y_box, axis=0)

    return features_box, y_box, false_mask_box, labeled_mask_box


def save_features(algorithm, epoch, source_prototypes, source_prototypes_y, target_features, target_y, target_false_mask, target_labeled_mask):
    # make dictionary for saving
    save_dict = {
        'source_prototypes': source_prototypes,
        'source_y': source_prototypes_y,
        'target_features': target_features,
        'target_y': target_y,
        'target_false_mask': target_false_mask,
        'target_labeled_mask': target_labeled_mask,

    }
    # save
    save_path = os.path.join(algorithm.save_dir, algorithm.save_name, 'features')
    save_file = os.path.join(save_path, 'features_{}.pickle'.format(epoch))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(file=save_file, mode='wb') as f:
        pickle.dump(save_dict, f)