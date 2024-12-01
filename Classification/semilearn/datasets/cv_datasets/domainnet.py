# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math

from .datasetbase import BasicDataset, BasicDatasetFromPath
from semilearn.datasets.utils import split_ssl_data, sample_train_val_data, sample_partial_data,\
            build_index, check_domain_name, get_transforms, print_pretraing_stats, print_fintuning_stats
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from torchvision import transforms



def get_domainnet(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):

    # Get img_path_list(=datas) and gt_list (=targets)
    data_dir = os.path.join(data_dir, name.lower())
    check_domain_name(args.src, args.trg)
    domain_name = args.src if args.src else args.trg
    label_file = os.path.join(data_dir, f"{domain_name}_list.txt")
    datas, targets = build_index(data_dir, label_file)
    
    img_size = args.img_size
    crop_size = int(math.floor(img_size * args.crop_ratio))

    transform_weak, transform_strong, transform_val = get_transforms(img_size, crop_size, alg)

    if args.src:
        # Split datas and targets into labeled and unlabeled data.
        datas, targets = np.array(datas), np.array(targets)
        total_data_size, train_ratio = len(datas), 0.8 # DomainNet has 20% validation data.
        train_idx, val_idx = sample_train_val_data(args, datas, targets, num_classes, train_ratio)
        train_datas = datas[train_idx]
        train_targets = targets[train_idx]
        val_datas =  datas[val_idx]
        val_targets = targets[val_idx]

        print_pretraing_stats(num_classes, train_targets, val_targets)

        assert args.algorithm == 'fullysupervised', 'Source pretrining must be running with the ''fullysupervised'' algorithm.'
        lb_dset = BasicDatasetFromPath(alg, train_datas, train_targets, num_classes, transform_weak, False, transform_strong, False)
        ulb_dset = BasicDatasetFromPath(alg, [], [], num_classes, transform_weak, False, transform_strong, False)
        eval_dset = BasicDatasetFromPath(alg, val_datas, val_targets,   num_classes, transform_val,  False, None, False)

    elif args.trg:
        if args.trg_eval_mode == 'finetune=valset':
            train_datas, train_targets = datas, targets
            val_datas, val_targets = datas, targets
        elif args.trg_eval_mode == 'finetune!=valset':
            datas, targets = np.array(datas), np.array(targets)
            train_idx, val_idx = sample_train_val_data(args, datas, targets, num_classes, train_ratio=0.8)
            train_datas = datas[train_idx]
            train_targets = targets[train_idx]
            val_datas =  datas[val_idx]
            val_targets = targets[val_idx]
        else:
            raise ValueError(f"Invalid trg_eval_mode: {args.trg_eval_mode}")

        # Split datas and targets into labeled and unlabeled data.
        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, train_datas, train_targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    
        print_pretraing_stats(num_classes, train_targets, val_targets)
        print_fintuning_stats(num_classes, lb_targets, ulb_targets)
        
        # With visda, domainnet, and officehome dataset, you can not conduct expeiement for remixmatch. 
        # If you want to running remixmatch, refer to 'semilearn/datasets/cv_datasets/svhn.py line 90~103'
        
        lb_dset = BasicDatasetFromPath(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, False)
        ulb_dset = BasicDatasetFromPath(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)
        eval_dset = BasicDatasetFromPath(alg, val_datas, val_targets, num_classes, transform_val, False, None, False)
    else:
        raise ValueError("Both source and target domain cannot be None.")

    return lb_dset, ulb_dset, eval_dset