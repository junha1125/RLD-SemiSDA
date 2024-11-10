import pickle
import logging
from glob import glob

import os,sys,inspect
import logging
import numpy as np
import argparse

import torchvision, torchvision.transforms
import sklearn, sklearn.model_selection
import sklearn.metrics
import copy
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import numpy as np
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

from tqdm import tqdm
from os.path import exists, join

try:
    import incl
    INCL_IMPORTED = True
except:
    INCL_IMPORTED = False

def print_fn(gpu, str):
    if gpu == 0:
        print(str)


def get_logger(name, save_path=None, level='INFO'):
    """
    create logger function
    """
    logger = logging.getLogger(name)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', level=getattr(logging, level))

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def SubsetDataset(dataset, inds):
    dataset = copy.deepcopy(dataset)
    dataset.labels = dataset.labels[inds]
    dataset.csv = dataset.csv.iloc[inds]
    return dataset


class DistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, num_samples=None, **kwargs):
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            else:
                rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.total_size = num_samples
        assert num_samples % self.num_replicas == 0, f'{num_samples} samples cant' \
                                                     f'be evenly distributed among {num_replicas} devices.'
        self.num_samples = int(num_samples // self.num_replicas)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        n = len(self.dataset)
        n_repeats = self.total_size // n
        n_remain = self.total_size % n
        indices = [torch.randperm(n, generator=g) for _ in range(n_repeats)]
        indices.append(torch.randperm(n, generator=g)[:n_remain])
        indices = torch.cat(indices, dim=0).tolist()

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def model_resume(cfg, model, logger):
    if cfg.resume_path:
        resume_path = join(cfg.resume_path)
        
        weights_files = glob(join(resume_path, f'{cfg.model}-e*.pt'))  
        epochs = np.array(
            [int(w[len(join(resume_path, f'{cfg.model}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
        recent_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
        
        best_file = join(resume_path, f'{cfg.model}-best.pt')
        
        load_file = best_file ## TODO
        saved_dict = torch.load(load_file)
        # remove 'module.' in state_dict
        new_state_dict = {}
        for k, v in saved_dict['model'].items():
            if k.startswith('module.'):
                k = k.replace('module.', '')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        
        opt_thres = saved_dict['opt_thres']

        with open(join(resume_path, f'{cfg.model}-status.pkl'), 'rb') as f:
            status = pickle.load(f)
        logger.info("Weights loaded: {0}".format(load_file))
        return model, opt_thres


@torch.no_grad()
def evaluation(cfg, epoch, iter, model, gpu, data_loader, criterion, logger, best_metric):
    model.eval()

    avg_loss = []
    task_outputs={}
    task_targets={}
    for task in range(data_loader.dataset.labels.shape[1]):
        task_outputs[task] = []
        task_targets[task] = []
        
    t = tqdm(data_loader) if gpu == 0 else data_loader
    for batch_idx, samples in enumerate(t):            
        images = samples["img"].to(gpu)
        targets = samples["lab"].to(gpu)

        outputs = model(images)
        
        loss = torch.zeros(1).to(gpu).double()
        for task in range(targets.shape[1]):
            task_output = outputs[:,task]
            task_target = targets[:,task]
            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]
            task_target = task_target[mask]
            if len(task_target) > 0:
                loss += criterion(task_output.double(), task_target.double())
            
            task_outputs[task].append(task_output.detach().cpu().numpy())
            task_targets[task].append(task_target.detach().cpu().numpy())

        loss = loss.sum()
        
        avg_loss.append(loss.detach().cpu().numpy())
        # print_fn(gpu, f'Epoch {epoch + 1} - Val - Loss = {np.mean(avg_loss):4.4f}')
        
    for task in range(len(task_targets)):
        task_outputs[task] = np.concatenate(task_outputs[task])
        task_targets[task] = np.concatenate(task_targets[task])

    task_aucs = []
    for task in range(len(task_targets)):
        if len(np.unique(task_targets[task]))> 1:
            task_auc = roc_auc_score(task_targets[task], task_outputs[task])
            task_aucs.append(task_auc)
        else:
            task_aucs.append(np.nan)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])

    add_dict = None
    if np.mean(auc) > best_metric:
        best_metric = np.mean(auc)
        add_dict = {f'best{task}' : task_aucs[task] for task in range(len(task_targets))}
        if gpu == 0:
            torch.save(model, join(cfg.save_path, f'{cfg.model}-best.pt'))
    
    all_auc_string = ""
    for task in range(len(task_targets)):
        all_auc_string += f"{task_aucs[task]:4.4f}, "    
    logger.info("Epoch: {} |AvgAUC: {:4.4f}|BestAUC: {:4.4f}| EveryAUC: {}".format(epoch, auc, best_metric, all_auc_string))
    
    if cfg.use_incl and INCL_IMPORTED:
        auc_dict = {'avg_auc': auc, 'best_auc': best_metric}
        if add_dict: auc_dict.update(add_dict)
        incl.log(auc_dict, step=iter)
    model.train()
    return best_metric