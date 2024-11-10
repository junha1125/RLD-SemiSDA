import os,sys,inspect
import warnings
warnings.filterwarnings("ignore")

from glob import glob
from os.path import exists, join
import matplotlib.pyplot as plt
import numpy as np
import pprint
import argparse
import copy
from tqdm import tqdm

import torch
import torchvision, torchvision.transforms
import skimage.transform
import sklearn, sklearn.model_selection
import sklearn.metrics
from sklearn.metrics import roc_auc_score, accuracy_score

import random
import torchxrayvision as xrv

import torch.multiprocessing as mp
import torch.distributed as dist

from utils import print_fn, get_logger, str2bool, SubsetDataset, DistributedSampler, evaluation, model_resume
from adapt_utils import get_thresholds_for_uda, get_indices_for_rld, get_label_indices


def main(cfg):
    if cfg.distributed:
        ngpus_per_node = torch.cuda.device_count()
        total_batch_size = cfg.batch_size * (1 + cfg.ulb_ratio + cfg.rld_num_append) 
        assert total_batch_size * ngpus_per_node == 128, "Total batch_size in all gpu must be 128"
        cfg.world_size = ngpus_per_node * cfg.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        main_worker(cfg.rank, cfg.world_size, cfg)


def main_worker(gpu, ngpus_per_node, cfg):
    """
    Args:
        gpu (int): GPU id to use.
        ngpus_per_node (int): Number of GPUs per node.
        cfg (dict): Configuration dictionary.
    Description:
        - Set random seed
        - Load dataset
        - Load model
        - Load optimizer, scheduler, criterion
        - Adapt the source model
    """

    # initialize training
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # set name for output directory
    output_name = "FB1" + "-lb" + str(cfg.num_lb_per_task) + "-seed" + str(cfg.seed)
    if cfg.eval_only:
        output_name = "eval-only"
    cfg.save_path = join(cfg.output_dir, output_name)
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path, exist_ok=True)

    # set logger
    level = "INFO" if gpu == 0 else "WARNING"
    logger = get_logger(output_name, save_path=cfg.save_path, level=level)
    logger.info(cfg)
    
    if cfg.distributed:
        cfg.rank = cfg.rank * ngpus_per_node + gpu  # compute global rank
        dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:"+cfg.ddp_url,
                                world_size=cfg.world_size, rank=cfg.rank)

    # define data augmentation
    data_train_aug = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.RandomAffine(degrees=cfg.train_aug_rot, 
                                            translate=(cfg.train_aug_trans, cfg.train_aug_trans), 
                                            scale=(1.0-cfg.train_aug_scale, 1.0+cfg.train_aug_scale)),
        torchvision.transforms.ToTensor()])
    data_eval_aug = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.ToTensor()])

    mimic_trg = xrv.datasets.MIMIC_224_Dataset(
            imgpath=join(cfg.dataset_dir, "files_224"),
            csvpath=join(cfg.dataset_dir, "mimic-cxr-2.0.0-chexpert.csv.gz"),
            metacsvpath=join(cfg.dataset_dir, "mimic-cxr-2.0.0-metadata.csv.gz"),
            transform=None, data_aug=data_train_aug, unique_patients=False, views=[cfg.trg])
    train_domain_dataset = mimic_trg
    
    base_dir = join(cfg.dataset_dir, 'train_val_indices')
    val_inds_name = join(base_dir, f"val_inds_{cfg.trg}.npy")
    train_inds_name = join(base_dir, f"train_inds_{cfg.trg}.npy")
    if exists(val_inds_name) and exists(train_inds_name):
        val_inds = np.load(val_inds_name)
        train_inds = np.load(train_inds_name)
        logger.info(f"Loaded indices from {base_dir}")
    else:
        gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.95,test_size=0.05, random_state=cfg.seed)
        train_inds, val_inds = next(gss.split(X=range(len(train_domain_dataset)), groups=train_domain_dataset.csv.patientid)) 
        np.save(val_inds_name, val_inds)
        np.save(train_inds_name, train_inds)
    logger.info(f"train_inds: {len(train_inds)}, val_inds: {len(val_inds)}")
    
    original_pathologies = train_domain_dataset.pathologies # sorted(['Cardiomegaly', 'Effusion'])
    relabel_pathologies = original_pathologies 
    xrv.datasets.relabel_dataset(relabel_pathologies, train_domain_dataset, silent=True)
    train_dataset = SubsetDataset(train_domain_dataset, train_inds) 
    val_dataset = SubsetDataset(train_domain_dataset, val_inds)     
    val_dataset.data_aug = data_eval_aug
    len_task = train_dataset.labels.shape[1]
    
    # define unlabeled dataset and dataloader (all data is used for unlabeled data)
    ulb_inds = np.array([i for i in range(len(train_dataset))])
    train_dataset_unlabeled = copy.deepcopy(train_dataset)

    train_sampler_unlabeled = DistributedSampler(dataset=train_dataset_unlabeled, 
                                                 num_replicas=cfg.world_size,
                                                 rank=gpu,
                                                 num_samples=len(ulb_inds)//cfg.world_size*cfg.world_size,
                                                 shuffle=True) \
                    if cfg.distributed else torch.utils.data.RandomSampler(train_dataset_unlabeled)
    train_loader_unlabeled = torch.utils.data.DataLoader(train_dataset_unlabeled,
                                                batch_size=cfg.batch_size*cfg.ulb_ratio,
                                                num_workers=cfg.workers, 
                                                sampler=train_sampler_unlabeled,)
    valid_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=512,
                                                num_workers=cfg.workers, 
                                                shuffle=False)
    dump_loader_unlabeled = torch.utils.data.DataLoader(train_dataset_unlabeled,
                                                batch_size=512,
                                                num_workers=cfg.workers, 
                                                shuffle=False)

    # define model, optimizer and load pretrained weights
    model = xrv.models.DenseNet(num_classes=len(original_pathologies), in_channels=1, # train_dataset.labels.shape[1]
                                    **xrv.models.get_densenet_params(cfg.model))
    model, opt_thres = model_resume(cfg, model, logger)

    # re-define classifier, refer to original_pathologies and relabel_pathologies
    # if all pathologies are used, then no need to re-define classifier (ignore this block)
    new_opt_thres = [0 for i in range(len(relabel_pathologies))]
    new_classifier = torch.nn.Linear(model.classifier.in_features, len(relabel_pathologies))
    for i, pathologie in enumerate(relabel_pathologies):
        idx = original_pathologies.index(pathologie)
        new_classifier.weight.data[i] = model.classifier.weight.data[idx]
        new_classifier.bias.data[i] = model.classifier.bias.data[idx]
        new_opt_thres[i] = opt_thres[idx]
    model.classifier = new_classifier
    opt_thres = new_opt_thres

    model.cuda(gpu)
    torch.cuda.set_device(gpu)
    if cfg.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    criterion = torch.nn.BCEWithLogitsLoss()

    if cfg.eval_only: 
        evaluation(cfg, 0, 0, model, gpu, valid_loader, criterion, logger, 0.0); return

    # define data_loaders for labeled data
    # FIXME: get_label_indices and get_thresholds_for_uda can be merged
    lb_inds = get_label_indices(cfg, train_dataset, model, gpu, opt_thres, dump_loader_unlabeled, logger)
    logger.info(f"lb_inds: {len(lb_inds)}, ulb_inds: {len(ulb_inds)}")

    train_dataset_labeled = SubsetDataset(train_dataset, lb_inds)

    train_sampler_labeled = DistributedSampler(dataset=train_dataset_labeled, 
                                                 num_replicas=cfg.world_size,
                                                 rank=gpu,
                                                 num_samples=len(ulb_inds)//cfg.world_size*cfg.world_size,
                                                 shuffle=True) \
                    if cfg.distributed else torch.utils.data.RandomSampler(train_dataset_labeled)
    train_loader_labeled = torch.utils.data.DataLoader(train_dataset_labeled,
                                                batch_size=cfg.batch_size,
                                                num_workers=cfg.workers, 
                                                sampler=train_sampler_labeled,)
    
    thresholds_uda, task_outputs, task_targets = get_thresholds_for_uda(cfg, model, opt_thres, gpu, dump_loader_unlabeled, logger, cal_acc=True) 
    if cfg.use_rld: rld_sample_inds = get_indices_for_rld(cfg, task_outputs, len_task, logger, task_targets)

    start_epoch = 0
    c_iter = 0
    best_metric = 0.0
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        if hasattr(train_loader_unlabeled.sampler, 'set_epoch'): 
            train_loader_labeled.sampler.set_epoch(epoch)
            train_loader_unlabeled.sampler.set_epoch(epoch)
        train_loader = zip(train_loader_labeled, train_loader_unlabeled)

        if cfg.taskweights:
            weights = np.nansum(train_loader_unlabeled.dataset.labels, axis=0)
            weights = weights.max() - weights + weights.mean()
            weights = weights/weights.max()
            weights = torch.from_numpy(weights).to(gpu).float()

        avg_loss = []
        t = tqdm(train_loader, total=len(train_loader_unlabeled)) if gpu == 0 else train_loader
        for batch_idx, (samples_lb, samples_ulb) in enumerate(t):
            c_iter += 1
            optimizer.zero_grad()
            scheduler.step()
            
            if not cfg.use_rld:
                loss = train_step(cfg, model, gpu, samples_lb, samples_ulb, criterion, weights, thresholds_uda)
            else:
                loss = train_step_with_rld(cfg, model, gpu, samples_lb, samples_ulb, criterion, \
                                           weights, thresholds_uda, rld_sample_inds, dump_loader_unlabeled)
            
            # here regularize the weight matrix when label_concat is used
            if cfg.label_concat_reg:
                if not cfg.label_concat:
                    raise Exception("cfg.label_concat must be true")
                weight = model.classifier.weight
                num_labels = len(xrv.datasets.default_pathologies)
                num_datasets = weight.shape[0]//num_labels
                weight_stacked = weight.reshape(num_datasets,num_labels,-1)
                label_concat_reg_lambda = torch.tensor(0.1).to(gpu).float()
                for task in range(num_labels):
                    dists = torch.pdist(weight_stacked[:,task], p=2).mean()
                    loss += label_concat_reg_lambda*dists
                    
            loss = loss.sum()
                
            if cfg.weightreg:
                loss += model.classifier.weight.abs().sum()
            
            loss.backward()
            optimizer.step()

            avg_loss.append(loss.detach().cpu().numpy())

            if c_iter % cfg.eval_iter == 0 or cfg.eval_iter == 1:
                logger.info(f'Validation at iter {c_iter} epoch {epoch} lr {scheduler.get_last_lr()[0]}')
                best_metric = evaluation(cfg, epoch, c_iter , model, gpu, valid_loader, criterion, logger, best_metric)
            if c_iter % cfg.get_thres_iter == 0 and c_iter > 0:
                thresholds_uda, task_outputs, task_targets = get_thresholds_for_uda(cfg, model, opt_thres, gpu, dump_loader_unlabeled, logger, cal_acc=False) 
                if cfg.use_rld: rld_sample_inds = get_indices_for_rld(cfg, task_outputs, len_task, logger, task_targets)


def train_step(cfg, model, gpu, samples_lb, samples_ulb, criterion, weights, thresholds_uda):
    """
    Performs a single training step using labeled and unlabeled data, applying supervised 
    and unsupervised losses for semi-supervised learning.
    """
    num_labels = len(samples_lb['idx'])
    samples = {key: torch.cat((samples_lb[key], samples_ulb[key]), dim=0) for key in samples_lb.keys()}

    images = samples["img"].float().to(gpu)
    targets = samples["lab"].to(gpu)

    outputs = model(images)
    outputs_lb, outputs_ulb = outputs[:num_labels], outputs[num_labels:]
    targets_lb, targets_ulb = targets[:num_labels], targets[num_labels:]
    
    loss = torch.zeros(1).to(gpu).float()
    
    # Supervised loss
    for task in range(targets.shape[1]):
        task_output = outputs_lb[:,task] 
        task_target = targets_lb[:,task] 
        mask = ~torch.isnan(task_target) 
        task_output = task_output[mask]  
        task_target = task_target[mask]  
        if len(task_target) > 0:         
            task_loss = criterion(task_output.float(), task_target.float())
            if cfg.taskweights:
                loss += weights[task]*task_loss
            else:
                loss += task_loss

    # Unsupervised loss
    if not cfg.label_only:
        for task in range(targets.shape[1]):
            task_output = outputs_ulb[:,task]    
            task_sigmoid_output = torch.sigmoid(task_output.detach())
            
            task_target = torch.full_like(task_sigmoid_output, np.nan)
            task_target[task_sigmoid_output > thresholds_uda[task][1]] = 1
            task_target[task_sigmoid_output < thresholds_uda[task][0]] = 0
            task_target = task_target.to(gpu)

            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]  
            task_target = task_target[mask]  
            if len(task_target) > 0:        
                task_loss = criterion(task_output.float(), task_target.float())
                if cfg.taskweights:
                    loss += weights[task]*task_loss
                else:
                    loss += task_loss
    return loss


def train_step_with_rld(cfg, model, gpu, samples_lb, samples_ulb, criterion, weights, thresholds_uda, rld_sample_inds, dump_loader_unlabeled):
    """
    Executes a training step with additional reliable labeled data (RLD), enhancing labeled data 
    with highly reliable samples for improved model robustness.
    """
    num_lb_origin = len(samples_lb['idx'])
    samples_lb = retrieval_latent_defending(cfg, samples_lb, rld_sample_inds, dump_loader_unlabeled)
    num_labels = len(samples_lb['idx'])
    lambda_lb = num_lb_origin / num_labels # Scale factor to maintain original labeled data influence after RLD augmentation

    samples = {key: torch.cat((samples_lb[key], samples_ulb[key]), dim=0) for key in samples_lb.keys()}

    images = samples["img"].float().to(gpu)
    targets = samples["lab"].to(gpu)

    outputs = model(images)
    outputs_lb, outputs_ulb = outputs[:num_labels], outputs[num_labels:]
    targets_lb, targets_ulb  = targets[:num_labels], targets[num_labels:]
    
    loss = torch.zeros(1).to(gpu).float()
    
    # Supervised loss
    for task in range(targets.shape[1]):
        task_output = outputs_lb[:,task] 
        task_target = targets_lb[:,task] 
        mask = ~torch.isnan(task_target) 
        task_output = task_output[mask]  
        task_target = task_target[mask]  
        if len(task_target) > 0:         
            task_loss = criterion(task_output.float(), task_target.float())
            if cfg.taskweights:
                loss += weights[task] * task_loss * lambda_lb
            else:
                loss += task_loss * lambda_lb

    # Unsupervised loss
    if not cfg.label_only:
        for task in range(targets.shape[1]):
            task_output = outputs_ulb[:,task]    
            task_sigmoid_output = torch.sigmoid(task_output.detach())
            
            task_target = torch.full_like(task_sigmoid_output, np.nan)
            task_target[task_sigmoid_output > thresholds_uda[task][1]] = 1
            task_target[task_sigmoid_output < thresholds_uda[task][0]] = 0
            task_target = task_target.to(gpu)

            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]  
            task_target = task_target[mask]  
            if len(task_target) > 0:        
                task_loss = criterion(task_output.float(), task_target.float())
                if cfg.taskweights:
                    loss += weights[task]*task_loss
                else:
                    loss += task_loss
    return loss


def retrieval_latent_defending(cfg, samples_lb, rld_sample_inds, dump_loader_unlabeled):
    """
    Enhances the labeled dataset by adding reliable samples based on the 
    Retrieval Latent Defending (RLD) method to improve model stability during training.

    Args:
        cfg: Configuration object containing various experiment settings.
        samples_lb (dict): Batch of labeled data containing 'img', 'lab', and 'idx' keys.
                           - 'img': Tensor of images.
                           - 'lab': Tensor of labels for each image, with NaN values for unknown labels.
                           - 'idx': Indices of each sample within the dataset.
        rld_sample_inds (ndarray): Array of reliable True Positive and True Negative samples based on pseudo labels
                                   for each task (pathology), structured as `[label=0 or 1][pathology]`.
        dump_loader_unlabeled (DataLoader): DataLoader object containing the unlabeled dataset.

    Returns:
        dict: The updated `samples_lb` dictionary with additional reliable samples included.
              - 'img': Tensor combining original images and the added reliable samples.
              - 'lab': Tensor combining original labels and the added labels. 
                       For added labels, only the selected task index has the value, others are NaN.
              - 'idx': Tensor combining original indices and the indices of added samples
    """
    
    dataset = dump_loader_unlabeled.dataset
    add_img_lb = []
    add_lab_lb = []
    add_idx_lb = []
    # target_lb = [] # for debugging
    for label in samples_lb['lab']:
        non_nan_indices = (~torch.isnan(label)).nonzero(as_tuple=False).squeeze().tolist()
        task = random.choice(non_nan_indices)
        target = int(label[task].item())
        _selected =  torch.randperm(len(rld_sample_inds[target, task]))[:cfg.rld_num_append]
        selected = rld_sample_inds[target, task][_selected]
        new_label = torch.full_like(label, float('nan'))
        new_label[task] = target
        for idx in selected:
            add_sample = dataset.__getitem__(idx)
            add_img_lb.append(add_sample['img'])
            add_lab_lb.append(new_label)
            add_idx_lb.append(add_sample['idx'])
            # target_lb.append(add_sample['lab'])
    
    add_img_lb = torch.stack(add_img_lb, dim=0)
    add_lab_lb = torch.stack(add_lab_lb, dim=0)
    add_idx_lb = torch.tensor(add_idx_lb, dtype=samples_lb['idx'].dtype)
    samples_lb['img'] = torch.cat((samples_lb['img'], add_img_lb), dim=0)
    samples_lb['lab'] = torch.cat((samples_lb['lab'], add_lab_lb), dim=0)
    samples_lb['idx'] = torch.cat((samples_lb['idx'], add_idx_lb), dim=0)
    return samples_lb


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="adapt")

    parser.add_argument('--trg', type=str, default="AP")
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--dataset', type=str, default="mimic")
    parser.add_argument('--dataset_dir', type=str, default="./data/mimic")
    parser.add_argument('--model', type=str, default="densenet121")
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--cuda', type=str2bool, default=True, help='')
    parser.add_argument('--num_epochs', type=int, default=10, help='')
    parser.add_argument('--batch_size', type=int, default=4, 
            help='the number of labeled data, not the total batch size\
            total batch size is batch_size*(ulb_ratio + cfg.rld_num_append)')  
    parser.add_argument('--ulb_ratio', type=int, default=7, help='')  
    parser.add_argument('--shuffle', type=str2bool, default=True, help='')
    parser.add_argument('--lr', type=float, default=1e-05, help='')
    parser.add_argument('--workers', type=int, default=1, help='')
    parser.add_argument('--taskweights', type=str2bool, default=True, help='')
    parser.add_argument('--featurereg', type=str2bool, default=False, help='')
    parser.add_argument('--weightreg', type=str2bool, default=False, help='')
    parser.add_argument('--data_aug', type=str2bool, default=True, help='')
    parser.add_argument('--train_aug_rot', type=int, default=15, help='')
    parser.add_argument('--train_aug_trans', type=float, default=0.1, help='')
    parser.add_argument('--train_aug_scale', type=float, default=0.15, help='')
    parser.add_argument('--label_concat', type=str2bool, default=False, help='')
    parser.add_argument('--label_concat_reg', type=str2bool, default=False, help='')
    parser.add_argument('--labelunion', type=str2bool, default=False, help='')
    parser.add_argument('--distributed', type=str2bool, default=True, help='')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--resume_path', type=str, default="outputs/mimic-densenet121-04PA_ddp2")
    parser.add_argument('--ddp_url', type=str, default="11112")
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--label_only', action='store_true', default=False)

    parser.add_argument('--num_lb_per_task', type=int, default=20, help='')   # for "random_feedback" and "negatively_biased_feedback"
    parser.add_argument('--num_lb_from_fal_neg', type=int, default=5, help='')   # for "random_feedback" and "negatively_biased_feedback"
    
    parser.add_argument('--label_select_strategy', type=str, default="random_feedback", choices=('random_feedback', 'negatively_biased_feedback'))

    parser.add_argument('--eval_iter', type=int, default=300)
    parser.add_argument('--get_thres_iter', type=int, default=500)

    parser.add_argument('--p_ratio', type=float, default=0.1, 
            help='Ratio for pseudo-labeling thresholds. Lower value means stricter pseudo-labeling. \
            Using p_ratio=0.1 selects top 90%% confident predictions as positive and bottom 10%% as negative.')
    parser.add_argument('--n_times', type=int, default=1, help='')  

    parser.add_argument('--use_rld', action='store_true', default=False)
    parser.add_argument('--num_reliable_rld', type=int, default=300, help='')  
    parser.add_argument('--rld_num_append', type=int, default=0)

    parser.add_argument('--serious_failure_ratio', type=float, default=0.7, 
            help='Ratio for selecting confident wrong predictions for labeling. Value of 0.7 means \
            selecting cases where model was very confident (top/bottom 70%%) but wrong, simulating radiologist feedback on serious mistakes.')

    return parser.parse_args()

if __name__ == '__main__':
    main(get_cfg())

"""
# positive_patientids = train_dataset_unlabeled.csv.iloc[positive_inds]['patientid'].values // (~train_dataset_unlabeled.csv['patientid'].isin(positive_patientids))).flatten()
"""