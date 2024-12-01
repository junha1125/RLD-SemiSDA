# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
from copy import deepcopy
from collections import Counter
import tqdm
from torch.utils.data.distributed import DistributedSampler

from semilearn.algorithms.hooks import MaskingHook
from semilearn.core.hooks import Hook
from semilearn.core.utils import get_data_loader

from .builder import AdaMoCo
from semilearn.core.utils import send_model_cuda
from semilearn.algorithms.utils import concat_all_gather, remove_wrap_arounds


class MocoBuildHook(Hook):
    """
    Build moco model and Generate pseudo-labels
    """
    def before_run(self, algorithm):
        algorithm.moco_model = AdaMoCo(
            algorithm.model,
            algorithm.ema_model,
            K=algorithm.queue_size,
            m=algorithm.m,
            T_moco=algorithm.T_moco,
            gpu=algorithm.args.gpu)

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
        self.pseudo_item_list, self.banks, eval_dict = eval_and_label_dataset(
                   self.data_loader_for_gen_pseudo, algorithm.moco_model, banks=None, args=algorithm.args)
        self.print_eval_and_label_dataset(algorithm, eval_dict)
        
    
    def after_train_epoch(self, algorithm):
        _, _, eval_dict = eval_and_label_dataset(
                   self.data_loader_for_gen_pseudo, algorithm.moco_model, banks=self.banks, args=algorithm.args)
        self.print_eval_and_label_dataset(algorithm, eval_dict)
        

    def print_eval_and_label_dataset(self, algorithm, eval_dict):
        current_epoch = 0 if not hasattr(algorithm, "epoch") else algorithm.epoch
        print_text = f"[MocoBuildHook] Iteration: {algorithm.it + 1}, Epoch: {current_epoch}, USE_EMA: {algorithm.ema_m != 0}, "
        for i, (key, item) in enumerate(eval_dict.items()):
            print_text += "{:s}: {:.2f}".format(key, item)  if not isinstance(item, str) else f"{key}: {item}"
            if i != len(eval_dict) - 1:
                print_text += ", "
            else:
                print_text += " "
        algorithm.print_fn(print_text)


def per_class_accuracy(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    acc_per_class = (matrix.diagonal() / matrix.sum(axis=1) * 100.0).round(2)
    # logging.info(
    #     f"Accuracy per class: {acc_per_class}, mean: {acc_per_class.mean().round(2)}"
    # )

    return acc_per_class


@torch.no_grad()
def eval_and_label_dataset(dataloader, moco_model, banks, args):
    eval_dict = dict()

    # make sure to switch to eval mode
    moco_model.eval()

    # run inference
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
        feats, logits_cls = moco_model(imgs, cls_only=True)

        features.append(feats)
        logits.append(logits_cls)
        gt_labels.append(labels)
        indices.append(idxs)

    features = torch.cat(features)
    logits = torch.cat(logits)
    gt_labels = torch.cat(gt_labels)
    indices = torch.cat(indices)

    if args.distributed and args.world_size > 1:
        # gather results from all ranks
        features = concat_all_gather(features)
        logits = concat_all_gather(logits)
        gt_labels = concat_all_gather(gt_labels)
        indices = concat_all_gather(indices)

        # remove extra wrap-arounds from DDP
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
    
    pred_labels = logits.argmax(dim=1)
    accuracy = (pred_labels == gt_labels).float().mean() * 100
    eval_dict["Model_acc"] = accuracy

    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: args.queue_size],
        "probs": probs[rand_idxs][: args.queue_size],
        "ptr": 0,
    }

    # refine predicted labels
    pred_labels, _, acc = refine_predictions(
        features, probs, banks, args=args, gt_labels=gt_labels
    )
    eval_dict["Refined_acc"] = acc

    pseudo_item_list = []
    for pred_label, idx in zip(pred_labels, indices):
        img_path = dataloader.dataset.data[idx]
        pseudo_item_list.append((str(img_path), int(pred_label)))

    return pseudo_item_list, banks, eval_dict


def get_distances(X, Y, dist_type="euclidean"):
    """
    Args:
        X: (N, D) tensor
        Y: (M, D) tensor
    """
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances


@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, args):
    pred_probs = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, args.dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs


@torch.no_grad()
def update_labels(banks, idxs, features, logits, args):
    # 1) avoid inconsistency among DDP processes, and
    # 2) have better estimate with more data points
    idxs = idxs if idxs.is_contiguous() else idxs.contiguous()
    features = features if features.is_contiguous() else features.contiguous()
    logits = logits if logits.is_contiguous() else logits.contiguous()
    if args.distributed:
        idxs = concat_all_gather(idxs)
        features = concat_all_gather(features)
        logits = concat_all_gather(logits)

    probs = F.softmax(logits, dim=1)

    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])


@torch.no_grad()
def refine_predictions(
    features,
    probs,
    banks,
    args,
    gt_labels=None,
):
    if args.refine_method == "nearest_neighbors":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        pred_labels, probs = soft_k_nearest_neighbors(
            features, feature_bank, probs_bank, args
        )
    elif args.refine_method is None:
        pred_labels = probs.argmax(dim=1)
    else:
        raise NotImplementedError(
            f"{args.refine_method} refine method is not implemented."
        )
    accuracy = None
    if gt_labels is not None:
        accuracy = (pred_labels == gt_labels).float().mean() * 100

    return pred_labels, probs, accuracy


@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100
    return accuracy


def info_nce_loss(logits_ins, pseudo_labels, mem_labels, contrast_type):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    # in class_aware mode, do not contrast with same-class samples
    if contrast_type == "class_aware" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    accuracy = calculate_acc(logits_ins, labels_ins)

    return loss


def classification_loss(logits_w, logits_s, target_labels, args):
    if args.ce_sup_type == "weak_weak":
        loss_cls = cross_entropy_loss(logits_w, target_labels, args)
        accuracy = calculate_acc(logits_w, target_labels)
    elif args.ce_sup_type == "weak_strong":
        loss_cls = cross_entropy_loss(logits_s, target_labels, args)
        accuracy = calculate_acc(logits_s, target_labels)
    else:
        raise NotImplementedError(
            f"{args.ce_sup_type} CE supervision type not implemented."
        )
    return loss_cls, accuracy


def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

    return loss_div


def diversification_loss(logits_w, logits_s, args):
    if args.ce_sup_type == "weak_weak":
        loss_div = div(logits_w)
    elif args.ce_sup_type == "weak_strong":
        loss_div = div(logits_s)
    else:
        loss_div = div(logits_w) + div(logits_s)

    return loss_div


def cross_entropy_loss(logits, labels, args):
    return F.cross_entropy(logits, labels)