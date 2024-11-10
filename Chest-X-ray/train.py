# %%
#!/usr/bin/env python
# coding: utf-8

import os,sys,inspect
import pickle
from glob import glob
from os.path import exists, join
import matplotlib.pyplot as plt
import numpy as np
import pprint
import argparse

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
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


def main(cfg):
    if cfg.distributed:
        ngpus_per_node = torch.cuda.device_count()
        assert cfg.batch_size * ngpus_per_node == 128, "Total batch_size in all gpu must be 128"
        cfg.world_size = ngpus_per_node * cfg.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        main_worker(cfg.rank, cfg.world_size, cfg)


def print_fn(gpu, str):
    if gpu == 0:
        print(str)


def main_worker(gpu, ngpus_per_node, cfg):
    """
    Args:
        gpu (int): GPU id to use.
        ngpus_per_node (int): Number of GPUs per node.
        cfg (argparse.Namespace): Configuration.
    Description:
        - Set random seed.
        - Load dataset.
        - Split dataset into train/val.
        - Load model.
        - Train model.
    """
    print_fn(gpu, cfg)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    if cfg.distributed:
        cfg.rank = cfg.rank * ngpus_per_node + gpu  # compute global rank
        dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:"+cfg.ddp_url,
                                world_size=cfg.world_size, rank=cfg.rank)

    data_train_aug = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.RandomAffine(degrees=cfg.train_aug_rot, 
                                            translate=(cfg.train_aug_trans, cfg.train_aug_trans), 
                                            scale=(1.0-cfg.train_aug_scale, 1.0+cfg.train_aug_scale)),
        torchvision.transforms.ToTensor()])
    data_eval_aug = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.ToTensor()])

    # load dataset
    # please check if the following folder and files exist
    mimic_src = xrv.datasets.MIMIC_224_Dataset(
            imgpath=join(cfg.dataset_dir, "files_224"),
            csvpath=join(cfg.dataset_dir, "mimic-cxr-2.0.0-chexpert.csv.gz"),
            metacsvpath=join(cfg.dataset_dir, "mimic-cxr-2.0.0-metadata.csv.gz"),
            transform=None, data_aug=data_train_aug, unique_patients=False, views=[cfg.src])
    domian = cfg.src
    train_domain_dataset = mimic_src    
    
    # split dataset
    base_dir = join(cfg.dataset_dir, 'train_val_indices')
    val_inds_name = join(base_dir, f"val_inds_{domian}.npy")
    train_inds_name = join(base_dir, f"train_inds_{domian}.npy")
    if exists(val_inds_name) and exists(train_inds_name):
        val_inds = np.load(val_inds_name)
        train_inds = np.load(train_inds_name)
        print_fn(gpu, f"Loaded indices from {base_dir}")
    else:
        gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.7,test_size=0.3, random_state=cfg.seed)
        train_inds, val_inds = next(gss.split(X=range(len(train_domain_dataset)), groups=train_domain_dataset.csv.patientid))
        np.save(val_inds_name, val_inds)
    print_fn(gpu, f"train_inds: {len(train_inds)}, val_inds: {len(val_inds)}")
    
    # subset dataset
    train_dataset = xrv.datasets.SubsetDataset(train_domain_dataset, train_inds)
    val_dataset = xrv.datasets.SubsetDataset(train_domain_dataset, val_inds)
    val_dataset.dataset.data_aug = data_eval_aug

    # define dataloader
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True) \
        if cfg.distributed else torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=cfg.batch_size,
                                                num_workers=cfg.workers, 
                                                sampler=train_sampler,)
    valid_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=512,
                                                num_workers=cfg.workers, 
                                                shuffle=False)

    # load model
    model = xrv.models.DenseNet(num_classes=train_dataset.labels.shape[1], in_channels=1, 
                                    **xrv.models.get_densenet_params(cfg.model))
    model.cuda(gpu)
    torch.cuda.set_device(gpu)
    if cfg.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5, amsgrad=True)
    criterion = torch.nn.BCEWithLogitsLoss()

    output_name = cfg.dataset + "-" + cfg.model + "-" + cfg.name
    output_dir = join(cfg.output_dir, output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # train model
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validauc = None
    auc_test = None
    metrics = []

    if cfg.resume_path:
        # load weights if resume_path exists
        resume_path = join(cfg.resume_path)
        weights_files = glob(join(resume_path, f'{cfg.model}-e*.pt'))  
        if len(weights_files):
            epochs = np.array(
                [int(w[len(join(resume_path, f'{cfg.model}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
            recent_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
            best_file = join(resume_path, f'{cfg.model}-best.pt')
            
            load_file = best_file ## TODO
            saved_dict = torch.load(load_file)
            model.load_state_dict(saved_dict['model'])
            opt_thres = saved_dict['opt_thres']
            print_fn(gpu, (type(opt_thres), opt_thres))

            with open(join(resume_path, f'{cfg.model}-status.pkl'), 'rb') as f:
                status = pickle.load(f)
            print_fn(gpu, "Weights loaded: {0}".format(load_file))
    
    # train
    for epoch in range(start_epoch, cfg.num_epochs):
        if not cfg.eval: avg_loss = train_epoch(cfg=cfg,
                                epoch=epoch,
                                model=model,
                                gpu=gpu,
                                optimizer=optim,
                                train_loader=train_loader,
                                criterion=criterion)
        
        results = valid_epoch(name='Valid',
                                        epoch=epoch,
                                        model=model,
                                        gpu=gpu,
                                        data_loader=valid_loader,
                                        criterion=criterion)

        opt_thres = calculate_optimal_thresholds(results, val_dataset.pathologies)

        if cfg.eval:             
            save_dict = {}
            if gpu == 0:
                save_dict['model'] = model.state_dict()
                save_dict['opt_thres'] = opt_thres
                torch.save(save_dict, join(output_dir, f'{cfg.model}-best.pt'))
            print_fn(gpu, "Evaluation done"); break
        
        auc_valid = results[0]
        if np.mean(auc_valid) > best_metric:
            best_metric = np.mean(auc_valid)
            save_dict = {}
            if gpu == 0:
                save_dict['model'] = model.state_dict()
                save_dict['opt_thres'] = opt_thres
                torch.save(save_dict, join(output_dir, f'{cfg.model}-best.pt'))
        
        status = {"epoch": epoch + 1, "trainloss": avg_loss, "validauc": auc_valid, 'best_metric': best_metric}
        metrics.append(status)
        if gpu == 0:
            with open(join(output_dir, f'{cfg.model}-status.pkl'), 'wb') as f:
                pickle.dump(metrics, f)
            save_dict = {}
            save_dict['model'] = model.state_dict()
            save_dict['opt_thres'] = opt_thres
            torch.save(save_dict, join(output_dir, f'{cfg.model}-e{epoch + 1}.pt'))


def train_epoch(cfg, epoch, model, gpu, train_loader, optimizer, criterion):
    model.train()

    if cfg.taskweights:
        weights = np.nansum(train_loader.dataset.labels, axis=0)
        weights = weights.max() - weights + weights.mean()
        weights = weights/weights.max()
        weights = torch.from_numpy(weights).to(gpu).float()
    
    avg_loss = []
    t = tqdm(train_loader) if gpu == 0 else train_loader
    for batch_idx, samples in enumerate(t):
        optimizer.zero_grad()
        
        images = samples["img"].float().to(gpu)
        targets = samples["lab"].to(gpu)

        outputs = model(images)
        
        loss = torch.zeros(1).to(gpu).float()
        for task in range(targets.shape[1]):
            task_output = outputs[:,task]
            task_target = targets[:,task]
            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]
            task_target = task_target[mask]
            if len(task_target) > 0:
                task_loss = criterion(task_output.float(), task_target.float())
                if cfg.taskweights:
                    loss += weights[task]*task_loss
                else:
                    loss += task_loss
        
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
        
        if cfg.featurereg:
            feat = model.features(images)
            loss += feat.abs().sum()
            
        if cfg.weightreg:
            loss += model.classifier.weight.abs().sum()

        loss.backward()
        optimizer.step()

        avg_loss.append(loss.detach().cpu().numpy())
        print_fn(gpu, f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')

    return np.mean(avg_loss)


def valid_epoch(name, epoch, model, gpu, data_loader, criterion):
    model.eval()

    avg_loss = []
    task_outputs={}
    task_targets={}
    for task in range(data_loader.dataset[0]["lab"].shape[0]):
        task_outputs[task] = []
        task_targets[task] = []
        
    with torch.no_grad():
        t = tqdm(data_loader) if gpu == 0 else data_loader
        for batch_idx, samples in enumerate(t):            
            images = samples["img"].to(gpu)
            targets = samples["lab"].to(gpu)

            outputs = model(images)
            
            loss = torch.zeros(1).to(gpu).double()
            for task in range(targets.shape[1]):
                task_output = outputs[:,task]
                task_output = torch.sigmoid(task_output)
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
            print_fn(gpu, f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')
            
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
    
        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task]))> 1:
                task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    all_auc_string = ""
    for task in range(len(task_targets)):
        all_auc_string += f"{task_aucs[task]:4.4f}, "    
    print_fn(gpu, f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f} | EveryAUC: {all_auc_string}')

    return auc, task_aucs, task_outputs, task_targets


def calculate_optimal_thresholds(results, pathologies):
    perf_dict = {}
    all_threshs = []
    all_min = []
    all_max = []
    all_ppv80 = []
    for i, patho in enumerate(pathologies):
        opt_thres = np.nan
        opt_min = np.nan
        opt_max = np.nan
        ppv80_thres = np.nan
        if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):
            
            #sigmoid
            all_outputs = results[2][i]
            
            fpr, tpr, thres = sklearn.metrics.roc_curve(results[3][i], all_outputs)
            pente = tpr - fpr
            opt_thres = thres[np.argmax(pente)]
            opt_min = all_outputs.min()
            opt_max = all_outputs.max()
            
            ppv, recall, thres = sklearn.metrics.precision_recall_curve(results[3][i], all_outputs)
            ppv80_thres_idx = np.where(ppv > 0.8)[0][0]
            ppv80_thres = thres[ppv80_thres_idx-1]
            
            auc = sklearn.metrics.roc_auc_score(results[3][i], all_outputs)
            
            # print(patho, auc, opt_thres)
            perf_dict[patho] = str(round(auc,2))
            
        else:
            perf_dict[patho] = "-"
            
        all_threshs.append(opt_thres)
        all_min.append(opt_min)
        all_max.append(opt_max)
        all_ppv80.append(ppv80_thres)
    print("Pathologies thresholds:", all_threshs)
    return all_threshs


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


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="04PA_ddp2")
    parser.add_argument('--src', type=str, default="PA")
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--dataset', type=str, default="mimic")
    parser.add_argument('--dataset_dir', type=str, default="./data/mimic")
    parser.add_argument('--model', type=str, default="densenet121")
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--cuda', type=str2bool, default=True, help='')
    parser.add_argument('--num_epochs', type=int, default=10, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')  
    parser.add_argument('--ulb_ratio', type=int, default=8, help='')  
    parser.add_argument('--shuffle', type=str2bool, default=True, help='')
    parser.add_argument('--lr', type=float, default=0.004, help='')
    parser.add_argument('--workers', type=int, default=4, help='')
    parser.add_argument('--taskweights', type=str2bool, default=True, help='')
    parser.add_argument('--featurereg', type=str2bool, default=False, help='')
    parser.add_argument('--weightreg', type=str2bool, default=False, help='')
    parser.add_argument('--data_aug', type=str2bool, default=True, help='')
    parser.add_argument('--train_aug_rot', type=int, default=45, help='')
    parser.add_argument('--train_aug_trans', type=float, default=0.15, help='')
    parser.add_argument('--train_aug_scale', type=float, default=0.15, help='')
    parser.add_argument('--label_concat', type=str2bool, default=False, help='')
    parser.add_argument('--label_concat_reg', type=str2bool, default=False, help='')
    parser.add_argument('--labelunion', type=str2bool, default=False, help='')

    parser.add_argument('--distributed', type=str2bool, default=True, help='')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--ddp_url', type=str, default="11102")
    # if --eval  exists, then only eval
    parser.add_argument('--eval', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    main(get_cfg())