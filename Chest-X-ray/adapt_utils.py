from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision, torchvision.transforms
import sklearn, sklearn.model_selection
import sklearn.metrics
from sklearn.metrics import roc_auc_score, accuracy_score

import torchxrayvision as xrv

from tqdm import tqdm

DEBUG = False


@torch.no_grad()
def get_task_outputs(cfg, model, opt_thres, gpu, data_loader, logger):
    """
    Evaluates model predictions and returns outputs, predictions, results, and targets for each task.
    """
    model.eval()
    indixes = []
    task_outputs={}
    task_targets={}
    task_preds={}
    task_pred_results = {}
    len_task = data_loader.dataset.labels.shape[1]

    for task in range(len_task):
        task_outputs[task] = []
        task_targets[task] = []
        task_preds[task] = []
        task_pred_results[task] = []
    
    t = tqdm(data_loader) if gpu == 0 else data_loader
    for batch_idx, samples in enumerate(t):            
        images = samples["img"].to(gpu)
        targets = samples["lab"].to(gpu)
        indixes.append(samples["idx"])

        outputs = model(images)

        for task in range(len_task):
            task_output = outputs[:,task]
            task_target = targets[:,task]
            task_output = torch.sigmoid(task_output)
            pred = torch.zeros_like(task_output)
            pred[opt_thres[task] < task_output] = 1
            task_outputs[task].append(task_output.detach().cpu().numpy())    
            task_targets[task].append(task_target.detach().cpu().numpy())
            task_preds[task].append(pred.detach().cpu().numpy())

    for task in range(len_task):
        task_outputs[task] = np.concatenate(task_outputs[task])
        task_targets[task] = np.concatenate(task_targets[task])
        task_preds[task] = np.concatenate(task_preds[task])
        task_pred_results[task] = task_preds[task] == task_targets[task] # bool list
        task_pred_results[task] = np.where(np.isnan(task_targets[task]), np.nan, task_pred_results[task]) # Correctness: 1(True, correct prediciton) or 0 or nan
    indixes = np.concatenate(indixes)
    assert np.all(indixes == np.arange(len(data_loader.dataset)))

    model.train()
    return task_outputs, task_preds, task_pred_results, task_targets


def get_thresholds_for_uda(cfg, model, opt_thres, gpu, dump_loader_unlabeled, logger, cal_acc=False):
    """
    Calculates confidence thresholds for unsupervised domain adaptation pseudo-labeling.
    """
    if cfg.label_only or cfg.eval_only:
        return None, None, None
    logger.info("Get thresholds for unsupervised adaptation")
    len_task = dump_loader_unlabeled.dataset.labels.shape[1]    
    task_outputs, task_preds, task_pred_results, task_targets = get_task_outputs(cfg, model, opt_thres, gpu, dump_loader_unlabeled, logger)
    if cal_acc:
        for task in range(len_task):
            task_pred_results[task] = task_preds[task] == task_targets[task] # bool list
            task_pred_results[task] = np.where(np.isnan(task_targets[task]), np.nan, task_pred_results[task]) # Correctness: 1(True, correct prediciton) or 0 or nan
            if DEBUG:
                pos_acc = np.logical_and(task_pred_results[task] == 1, task_targets[task] == 1).sum() / (task_targets[task]==1).sum()
                neg_acc = np.logical_and(task_pred_results[task] == 1, task_targets[task] == 0).sum() / (task_targets[task]==0).sum()
                logger.info("[Source] {}: pos_acc {:0.2f} | neg_acc {:0.2f}".format(task, pos_acc*100, neg_acc*100))

    ## for each task(pathology), get the threshold for pseudo-labeling
    positive_thres = [1 for i in range(len_task)]
    negative_thres = [0 for i in range(len_task)]
    for i in range(len_task): 
        sorted_output = sorted(task_outputs[i])
        positive_thres[i] = sorted_output[int(len(task_outputs[i])*(1-cfg.p_ratio))]
        negative_thres[i] = sorted_output[int(len(task_outputs[i])*(cfg.p_ratio*cfg.n_times))]
        if DEBUG:
            pos_pseudo_samples = task_outputs[i] > positive_thres[i]
            neg_pseudo_samples = task_outputs[i] < negative_thres[i]
            precision = np.logical_and(pos_pseudo_samples, task_targets[i] == 1).sum() / pos_pseudo_samples.sum()
            recall = np.logical_and(neg_pseudo_samples, task_targets[i] == 0).sum() / neg_pseudo_samples.sum()
            pos_acc = np.logical_and(pos_pseudo_samples, task_targets[i] == 1).sum() / np.logical_and(pos_pseudo_samples, ~np.isnan(task_targets[i])).sum()
            neg_acc = np.logical_and(neg_pseudo_samples, task_targets[i] == 0).sum() / np.logical_and(neg_pseudo_samples, ~np.isnan(task_targets[i])).sum()  
            logger.info("{}: num_positive {:0.2f} | num_negative {:0.2f}".format(i, pos_pseudo_samples.sum(), neg_pseudo_samples.sum()))
            logger.info("[UDA] {}: pos_acc {:0.2f} precision {:0.2f} || neg_acc {:0.2f} recall {:0.2f}".format(i, pos_acc*100, precision*100, neg_acc*100,  recall*100))
    return np.stack([negative_thres, positive_thres], axis=1), task_outputs, task_targets


def get_indices_for_rld(cfg, task_outputs, len_task, logger, task_targets=None):
    """
    Selects samples for Retrieval Latent Defending (RLD) based on model confidence.
    For each task, identifies the most reliable True Positive (TP) and True Negative (TN) samples
    """
    if cfg.label_only or cfg.eval_only:
        return None
    logger.info("Get indices for retrieval latent defending")

    # Calculate thresholds
    positive_thres = [1 for i in range(len_task)]
    negative_thres = [0 for i in range(len_task)]
    for i in range(len_task): 
        sorted_output = sorted(task_outputs[i])
        positive_thres[i] = sorted_output[-(cfg.num_reliable_rld+1)]  
        negative_thres[i] = sorted_output[cfg.num_reliable_rld]
        if DEBUG:
            if task_targets is not None:
                pos_pseudo_samples = task_outputs[i] > positive_thres[i]
                neg_pseudo_samples = task_outputs[i] < negative_thres[i]
                precision = np.logical_and(pos_pseudo_samples, task_targets[i] == 1).sum() / pos_pseudo_samples.sum()
                recall = np.logical_and(neg_pseudo_samples, task_targets[i] == 0).sum() / neg_pseudo_samples.sum()
                pos_acc = np.logical_and(pos_pseudo_samples, task_targets[i] == 1).sum() / np.logical_and(pos_pseudo_samples, ~np.isnan(task_targets[i])).sum()
                neg_acc = np.logical_and(neg_pseudo_samples, task_targets[i] == 0).sum() / np.logical_and(neg_pseudo_samples, ~np.isnan(task_targets[i])).sum()  
                logger.info("{}: num_positive {:0.2f} | num_negative {:0.2f}".format(i, pos_pseudo_samples.sum(), neg_pseudo_samples.sum()))
                logger.info("[rld] {}: pos_acc {:0.2f} precision {:0.2f} || neg_acc {:0.2f} recall {:0.2f}".format(i, pos_acc*100, precision*100, neg_acc*100,  recall*100))

    # Get indices
    neg_inds_rld = [0 for i in range(len_task)]
    pos_inds_rld = [1 for i in range(len_task)]
    for i in range(len_task):
        neg_inds_rld[i] = np.where(task_outputs[i] < negative_thres[i])[0]
        pos_inds_rld[i] = np.where(task_outputs[i] > positive_thres[i])[0]

    # make neg_inds_rlds and pos_inds_rlds have the same length using min
    min_len = min([len(neg_inds_rld[i]) for i in range(len_task)] + [len(pos_inds_rld[i]) for i in range(len_task)])
    for i in range(len_task):
        neg_inds_rld[i] = neg_inds_rld[i][:min_len]
        pos_inds_rld[i] = pos_inds_rld[i][:min_len]
    
    bal_sample_inds = np.stack([neg_inds_rld, pos_inds_rld], axis=0) # [pred_label=0 or 1][pathology] # debug: task_outputs[0][bal_sample_inds[0][0]].mean()
    return bal_sample_inds


def get_label_indices(cfg, train_dataset, model, gpu, opt_thres, dump_loader_unlabeled, logger):
    """
    Selects indices for labeled data based on specified strategy.
    Supports "balanced_random" and "negatively_biased_feedback" selection methods.
    """
    logger.info("Get label indices under {} assumption".format(cfg.label_select_strategy))
    lb_inds_name = os.path.join(cfg.output_dir, 'lb_inds', f"lb_inds_{cfg.trg}_seed{cfg.seed}.npy")
    len_task = train_dataset.labels.shape[1]
    
    if os.path.exists(lb_inds_name):
        lb_inds = np.load(lb_inds_name)
        logger.info(f"Loaded label indices from {lb_inds_name}")
    else:
        if cfg.label_select_strategy == "balanced_random":
            lb_inds = []
            for task in range(len_task): # length of pathology
                task_targets = dump_loader_unlabeled.dataset.labels[:,task]
                
                positive_inds = np.argwhere(task_targets==1).flatten()
                positive_inds = np.random.choice(positive_inds, cfg.num_lb_per_task, replace=False)
                negative_inds = np.argwhere(task_targets==0).flatten()
                negative_inds = np.random.choice(negative_inds, cfg.num_lb_per_task, replace=False)

                lb_inds.append(np.concatenate([positive_inds, negative_inds]))
            lb_inds = np.concatenate(lb_inds)
        
        elif cfg.label_select_strategy == "negatively_biased_feedback":
            lb_inds = []
            task_outputs, task_preds, task_pred_results, task_targets = get_task_outputs(cfg, model, opt_thres, gpu, dump_loader_unlabeled, logger)
            for task in range(len_task): # length of pathology
                serious_failure_ratio = cfg.serious_failure_ratio # confident errors

                # FIXME each pathology has different serious_failure_ratio because different pathologies show varying the number of false samples.
                # Specifically:
                # - task 9 requires ratio=1.0 since it has very few false predictions, so we need to consider all false samples to get enough NBF samples (20)
                if task == 9:
                    serious_failure_ratio = 1.0
                if (task == 6 or task == 5) and cfg.serious_failure_ratio < 0.2:
                    serious_failure_ratio = cfg.serious_failure_ratio*2

                sigmoid_outputs = task_outputs[task]
                # predicted as normal(0) but actually abnormal(1). 
                false_0_inds = np.logical_and(task_preds[task] == 0, task_pred_results[task] == False)
                dump_thres = sorted(sigmoid_outputs[false_0_inds])[int(len(sigmoid_outputs[false_0_inds])*serious_failure_ratio)-1]
                false_0_inds = np.logical_and(false_0_inds, sigmoid_outputs < dump_thres)
                false_0_inds = np.where(false_0_inds)[0]
                positive_inds = np.random.choice(false_0_inds, cfg.num_lb_per_task, replace=False)
                # predicted as abnormal(1) but actually normal(0). 
                false_1_inds = np.logical_and(task_preds[task] == 1, task_pred_results[task] == False)
                dump_thres = sorted(sigmoid_outputs[false_1_inds])[int(len(sigmoid_outputs[false_1_inds])*(1-serious_failure_ratio))]
                false_1_inds = np.logical_and(false_1_inds, sigmoid_outputs > dump_thres)
                false_1_inds = np.where(false_1_inds)[0]
                negative_inds = np.random.choice(false_1_inds, cfg.num_lb_per_task, replace=False)
                
                lb_inds.append(np.concatenate([positive_inds, negative_inds]))
                logger.info(f"[NBF] ourlier rate {cfg.serious_failure_ratio} Num of serious failure of {task}: pred0_{len(false_0_inds)}, pred1_{len(false_1_inds)}")
            lb_inds = np.concatenate(lb_inds)
        else:
            raise Exception("cfg.label_select_strategy is not defined")

        os.makedirs(os.path.dirname(lb_inds_name), exist_ok=True)
        np.save(lb_inds_name, lb_inds) # next time, we can load indices of labeled samples without re-computation
    return lb_inds