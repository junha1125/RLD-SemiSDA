# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import random
import numpy as np
from PIL import ImageFilter
import torch
from torch.utils.data import sampler, DataLoader
import torch.distributed as dist
from torchvision import transforms
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from io import BytesIO

# TODO: better way
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def split_ssl_data(args, data, targets, num_classes,
                   lb_num_labels, ulb_num_labels=None,
                   lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0,
                   lb_index=None, ulb_index=None, include_lb_to_ulb=True):
    """
    data & target is splitted into labeled and unlabeled data.
    
    Args
        data: data to be split to labeled and unlabeled 
        targets: targets to be split to labeled and unlabeled 
        num_classes: number of total classes
        lb_num_labels: number of labeled samples. 
                       If lb_imbalance_ratio is 1.0, lb_num_labels denotes total number of samples.
                       Otherwise it denotes the number of samples in head class.
        ulb_num_labels: similar to lb_num_labels but for unlabeled data.
                        default to None, denoting use all remaining data except for labeled data as unlabeled set
        lb_imbalance_ratio: imbalance ratio for labeled data
        ulb_imbalance_ratio: imbalance ratio for unlabeled data
        lb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        ulb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeled data
    """
    data, targets = np.array(data), np.array(targets)
    lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, data, targets, num_classes, 
                                                    lb_num_labels, ulb_num_labels,
                                                    lb_imbalance_ratio, ulb_imbalance_ratio)
    
    # manually set lb_idx and ulb_idx, do not use except for debug
    if lb_index is not None:
        lb_idx = lb_index
    if ulb_index is not None:
        ulb_idx = ulb_index

    ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)
    ulb_idx = np.sort(ulb_idx)
    
    return data[lb_idx], targets[lb_idx], data[ulb_idx], targets[ulb_idx]


def sample_labeled_data():
    pass

def get_trg_eval_mode_type(trg_eval_mode):
    if trg_eval_mode == "finetune=valset": 
        return 1
    elif trg_eval_mode == "finetune!=valset":
        return 2
    else:
        raise ValueError("trg_eval_mode {} not supported".format(trg_eval_mode))

def sample_labeled_unlabeled_data(args, data, target, num_classes,
                                  lb_num_labels, ulb_num_labels=None,
                                  lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    dump_dir = os.path.join(args.data_dir, args.dataset, 'labeled_idx')
    os.makedirs(dump_dir, exist_ok=True)
    src_domain = os.path.basename(args.src_model_path).split('_')[0]
    lb_dump_path = os.path.join(dump_dir, f'lb_labels{lb_num_labels}_{src_domain[0]}2{args.trg[0]}_seed{args.seed}_idx.npy')
    # lb_dump_path = os.path.join(dump_dir, f'lb_labels{lb_num_labels}_{args.trg}_seed{args.seed}_e2_idx.npy')

    if os.path.exists(lb_dump_path):
        lb_idx = np.load(lb_dump_path)
        ulb_idx = np.array([i for i in range(len(target)) if i not in lb_idx])
        return lb_idx, ulb_idx 
    else:
        raise ValueError("labeled/unlabeled data dump does not exist in {}".format(lb_dump_path))
        # get samples per class
        if lb_imbalance_ratio == 1.0:
            # balanced setting, lb_num_labels is total number of labels for labeled data
            assert lb_num_labels % num_classes == 0, "lb_num_labels must be dividable by num_classes in balanced setting"
            lb_samples_per_class = [int(lb_num_labels / num_classes)] * num_classes
        else:
            # imbalanced setting, lb_num_labels is the maximum number of labels for class 1
            lb_samples_per_class = make_imbalance_data(lb_num_labels, num_classes, lb_imbalance_ratio)


        if ulb_imbalance_ratio == 1.0:
            # balanced setting
            if ulb_num_labels is None or ulb_num_labels == 'None':
                pass # ulb_samples_per_class = [int(len(data) / num_classes) - lb_samples_per_class[c] for c in range(num_classes)] # [int(len(data) / num_classes) - int(lb_num_labels / num_classes)] * num_classes
            else:
                assert ulb_num_labels % num_classes == 0, "ulb_num_labels must be dividable by num_classes in balanced setting"
                ulb_samples_per_class = [int(ulb_num_labels / num_classes)] * num_classes
        else:
            # imbalanced setting
            assert ulb_num_labels is not None, "ulb_num_labels must be set set in imbalanced setting"
            ulb_samples_per_class = make_imbalance_data(ulb_num_labels, num_classes, ulb_imbalance_ratio)

        lb_idx = []
        ulb_idx = []
        
        for c in range(num_classes):
            idx = np.where(target == c)[0]
            np.random.shuffle(idx)
            lb_idx.extend(idx[:lb_samples_per_class[c]])
            if ulb_num_labels is None or ulb_num_labels == 'None':
                ulb_idx.extend(idx[lb_samples_per_class[c]:])
            else:
                ulb_idx.extend(idx[lb_samples_per_class[c]:lb_samples_per_class[c]+ulb_samples_per_class[c]])
        
        if isinstance(lb_idx, list):
            lb_idx = np.asarray(lb_idx)
        if isinstance(ulb_idx, list):
            ulb_idx = np.asarray(ulb_idx)

        np.save(lb_dump_path, lb_idx)
        assert False, "we only use this function for saving labeled/unlabeled data"
        
        return lb_idx, ulb_idx


def sample_train_val_data(args, data, target, num_classes, train_ratio, file_name=None):
    '''
    samples for train data
    '''
    train_samples_per_class = [int(np.sum(target == c) * train_ratio) for c in range(num_classes)]

    train_idx = []
    val_idx = []
    
    if args.src:
        dump_dir = os.path.join(base_dir, 'data', args.dataset, 'pretraining_data_idx')
        os.makedirs(dump_dir, exist_ok=True)
        pv_dump_path = os.path.join(dump_dir, f'pretraining_{args.src}_seed{args.seed}_idx.npy')
    elif args.trg:
        dump_dir = os.path.join(args.data_dir, 'valset')
        file_name = f'{args.trg}_valset.npy' if file_name is None else file_name
        pv_dump_path = os.path.join(dump_dir, file_name)
        assert os.path.exists(pv_dump_path), "valset does not exist in {}".format(pv_dump_path)
    else:
        raise ValueError("Both source and target domain cannot be None.")

    if os.path.exists(pv_dump_path):
        val_idx = np.load(pv_dump_path)
        train_idx = np.array([i for i in range(len(target)) if i not in val_idx])
        return train_idx, val_idx
    else:
        for c in range(num_classes):
            idx = np.where(target == c)[0]
            np.random.shuffle(idx)
            train_idx.extend(idx[:train_samples_per_class[c]])
            val_idx.extend(idx[train_samples_per_class[c]:])
        
        if isinstance(train_idx, list):
            train_idx = np.asarray(train_idx)
        if isinstance(val_idx, list):
            val_idx = np.asarray(val_idx)
        np.save(pv_dump_path, val_idx)
        return train_idx, val_idx


def sample_partial_data(args, data, targets):
    """
    samples for partial data
    
    Args
        args.partial_ratio: ratio of partial data
    """
    data, targets = np.array(data), np.array(targets)

    dump_dir = os.path.join(base_dir, 'data', args.dataset, 'partial_data_idx')
    os.makedirs(dump_dir, exist_ok=True)
    pd_dump_path = os.path.join(dump_dir, f'partial_data{args.partial_ratio * 100}_seed{args.seed}_idx.npy')

    if os.path.exists(pd_dump_path):
        partial_idx = np.load(pd_dump_path)
    else:
        assert 0. <= args.partial_ratio <= 1.0, "partial ratio must be in [0, 1.0]"
        num_samples = len(targets)
        num_partial_samples = int(num_samples * args.partial_ratio)
        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        partial_idx = idx[:num_partial_samples]
        np.save(pd_dump_path, partial_idx)
    
    return data[partial_idx], targets[partial_idx]


def make_imbalance_data(max_num_labels, num_classes, gamma):
    """
    calculate samplers per class for imbalanced data
    """
    mu = np.power(1 / abs(gamma), 1 / (num_classes - 1))
    samples_per_class = []
    for c in range(num_classes):
        if c == (num_classes - 1):
            samples_per_class.append(int(max_num_labels / abs(gamma)))
        else:
            samples_per_class.append(int(max_num_labels * np.power(mu, c)))
    if gamma < 0:
        samples_per_class = samples_per_class[::-1]
    return samples_per_class


def get_collactor(args, net):
    if net == 'bert_base_uncased':
        from semilearn.datasets.collactors import get_bert_base_uncased_collactor
        collact_fn = get_bert_base_uncased_collactor(args.max_length)
    elif net == 'bert_base_cased':
        from semilearn.datasets.collactors import get_bert_base_cased_collactor
        collact_fn = get_bert_base_cased_collactor(args.max_length)
    elif net == 'wave2vecv2_base':
        from semilearn.datasets.collactors import get_wave2vecv2_base_collactor
        collact_fn = get_wave2vecv2_base_collactor(args.max_length_seconds, args.sample_rate)
    elif net == 'hubert_base':
        from semilearn.datasets.collactors import get_hubert_base_collactor
        collact_fn = get_hubert_base_collactor(args.max_length_seconds, args.sample_rate)
    else:
        collact_fn = None
    return collact_fn


def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = random.randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_transforms(img_size, crop_size, algrihtm):
    """
    Get transformation functions for each domain.
    This function is only used in 'visda.py', 'domainnet.py', 'officehome.py'.
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_weak = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if algrihtm in ['guidingpl','contratta']:
        transform_strong = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform_strong = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 10), # Note that (3, 5) was used for CIFAR, STL10, and SVHN, but (3, 10) for ImageNet.
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform_weak, transform_strong, transform_val


def print_pretraing_stats(num_classes, train_targets, val_targets):
    # Check the size of train and validation data.
    train_count = [0 for _ in range(num_classes)]
    val_count = [0 for _ in range(num_classes)]
    for c in train_targets:
        train_count[c] += 1
    for c in val_targets:
        val_count[c] += 1
    print("Number of samples per class for source pretraining")
    print("Train count: {}".format(train_count))
    print("Validation count: {}".format(val_count))


def print_fintuning_stats(num_classes, lb_targets, ulb_targets):
    # Split datas and targets into labeled and unlabeled data.
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))


def build_index(image_root: str, label_file: str):
    """Build a list of <image path, class label> items.
    Args:
        label_file: path to the domain-net label file
    Returns:
        item_list: a list of <image path, class label> items.
    """
    # read in items; each item takes one line
    with open(label_file, "r") as fd:
        lines = fd.readlines()
    lines = [line.strip() for line in lines if line]

    img_path_list = []
    gt_list = []
    for item in lines:
        img_file, label = item.split()
        img_path = os.path.join(image_root, img_file)
        label = int(label)
        img_path_list.append(img_path)
        gt_list.append(label)

    return img_path_list, gt_list


def check_domain_name(src_arg, trg_arg):
    if src_arg is None and trg_arg is None:
        raise ValueError("Both source and target domain cannot be None.")
    elif src_arg is not None and trg_arg is not None:
        raise ValueError("Both source and target domain cannot be defined.")
    else:
        pass

