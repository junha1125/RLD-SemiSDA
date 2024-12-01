# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import contextlib
import numpy as np
from tqdm import tqdm
from inspect import signature
from collections import OrderedDict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from semilearn.core.hooks import Hook, get_priority, CheckpointHook, TimerHook, LoggingHook, DistSamplerSeedHook, ParamUpdateHook, \
                                 EvaluationHook, EMAHook, WANDBHook, AimHook, INCLHook, RedefineDataHook, SaveFeatureHook
from semilearn.core.utils import get_dataset, get_data_loader, get_optimizer, get_cosine_schedule_with_warmup, Bn_Controller
from semilearn.core.criterions import CELoss, ConsistencyLoss
from semilearn.algorithms.utils import concat_all_gather, remove_wrap_arounds


class AlgorithmBase:
    """
        Base class for algorithms
        init algorithm specific parameters and common parameters
        
        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
    """
    def __init__(
        self,
        args,
        net_builder,
        tb_log=None,
        logger=None,
        **kwargs):
        
        # common arguments
        self.args = args
        self.num_classes = args.num_classes
        self.ema_m = args.ema_m
        self.epochs = args.epoch
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_log_iter = args.num_log_iter
        self.num_save_epoch = args.num_save_epoch
        self.num_iter_per_epoch = int(self.num_train_iter // self.epochs)
        self.lambda_u = args.ulb_loss_ratio 
        self.use_cat = args.use_cat
        self.use_amp = args.amp
        self.clip_grad = args.clip_grad
        self.save_name = args.save_name
        self.save_dir = args.save_dir
        self.resume = args.resume
        self.algorithm = args.algorithm

        # commaon utils arguments
        self.tb_log = tb_log
        self.print_fn = print if logger is None else logger.info
        self.ngpus_per_node = torch.cuda.device_count()
        self.loss_scaler = GradScaler()
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.gpu = args.gpu
        self.rank = args.rank
        self.distributed = args.distributed
        self.world_size = args.world_size

        # common model related parameters
        self.it = 0
        self.start_epoch = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.bn_controller = Bn_Controller()
        self.net_builder = net_builder
        self.ema = None

        # build dataset
        self.dataset_dict = self.set_dataset()

        # build data loader
        self.loader_dict = self.set_data_loader()

        # cv, nlp, speech builder different arguments
        self.model = self.set_model()
        self.ema_model = self.set_ema_model()

        # build optimizer and scheduler
        self.optimizer, self.scheduler = self.set_optimizer()

        # build supervised loss and unsupervised loss
        self.ce_loss = CELoss()
        self.consistency_loss = ConsistencyLoss()

        # other arguments specific to the algorithm
        # self.init(**kwargs)

        # set common hooks during training
        self._hooks = []  # record underlying hooks 
        self.hooks_dict = OrderedDict() # actual object to be used to call hooks
        self.set_hooks()
        self.log_dict = {}


    def init(self, **kwargs):
        """
        algorithm specific init function, to add parameters into class
        """
        raise NotImplementedError
    

    def set_dataset(self):
        """
        set dataset_dict
        """
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        # TODO: hard code here, need to be changed
        if self.args.num_labels == 0:
            self.print_fn("Note that the function ''train_step'' should not use labeled data, since you set num_labels as 0.") 
            num_label_dump = 120 if self.args.dataset == 'visda' else 1260
        else:
            num_label_dump = self.args.num_labels
        dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, num_label_dump, self.args.num_classes, self.args.data_dir, self.args.include_lb_to_ulb)
        if dataset_dict is None:
            return dataset_dict
        
        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.args.eval_dest_len = len(dataset_dict['eval'])
        self.print_fn("unlabeled data number: {}, labeled data number: {} | eval data number: {}".format(self.args.ulb_dest_len, self.args.lb_dest_len, self.args.eval_dest_len))
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict


    def set_data_loader(self):
        """
        set loader_dict
        """
        if self.dataset_dict is None:
            return
            
        self.print_fn("Create train and test data loaders")
        loader_dict = {}
        loader_dict['train_lb'] = get_data_loader(self.args,
                                                  self.dataset_dict['train_lb'],
                                                  self.args.batch_size,
                                                  data_sampler=self.args.train_sampler,
                                                  num_iters=self.num_train_iter,
                                                  num_epochs=self.epochs,
                                                  num_workers=self.args.num_workers,
                                                  distributed=self.distributed)

        loader_dict['train_ulb'] = get_data_loader(self.args,
                                                   self.dataset_dict['train_ulb'],
                                                   self.args.batch_size * self.args.uratio,
                                                   data_sampler=self.args.train_sampler,
                                                   num_iters=self.num_train_iter,
                                                   num_epochs=self.epochs,
                                                   num_workers=2 * self.args.num_workers,
                                                   distributed=self.distributed)

        eval_sampler = DistributedSampler(self.dataset_dict['eval'], shuffle=False) \
                                if (self.distributed and self.world_size > 1) else None
        loader_dict['eval'] = get_data_loader(self.args,
                                              self.dataset_dict['eval'],
                                              self.args.eval_batch_size,
                                              # make sure data_sampler is None for evaluation
                                              data_sampler=eval_sampler,
                                              num_workers=self.args.num_workers,
                                              distributed=self.distributed,
                                              drop_last=False)
        
        if self.dataset_dict['test'] is not None:
            test_sampler = DistributedSampler(self.dataset_dict['test'], shuffle=False) \
                                if (self.distributed and self.world_size > 1) else None
            loader_dict['test'] =  get_data_loader(self.args,
                                                   self.dataset_dict['test'],
                                                   self.args.eval_batch_size,
                                                   # make sure data_sampler is None for evaluation
                                                   data_sampler=test_sampler,
                                                   num_workers=self.args.num_workers,
                                                   distributed=self.distributed,
                                                   drop_last=False)
        self.print_fn(f'[!] data loader keys: {loader_dict.keys()}')
        return loader_dict

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, \
                                  self.args.layer_decay, net_name=self.args.net, linear_probing=self.args.linear_probing)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    self.num_train_iter,
                                                    num_warmup_steps=self.args.num_warmup_iter)
        return optimizer, scheduler

    def set_model(self):
        """
        initialize model
        """
        model = self.net_builder(pretrained=self.args.use_pretrain, 
                                 pretrained_path=self.args.pretrain_path,
                                # TODO: args=self.args, 
                                 num_classes=self.num_classes)
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = self.net_builder(pretrained=self.args.use_pretrain, 
                                 pretrained_path=self.args.pretrain_path,
                                # TODO: args=self.args, 
                                 num_classes=self.num_classes)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_hooks(self):
        """
        register necessary training hooks
        """
        # parameter update hook is called inside each train_step
        self.register_hook(ParamUpdateHook(), None, "HIGHEST")
        self.register_hook(EMAHook(), None, "HIGH")
        if self.args.trg_eval_src or self.args.save_feature or self.args.negatively_biased_feedback:
            self.register_hook(RedefineDataHook(), None, "HIGH")
        self.register_hook(EvaluationHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")
        if self.args.save_feature:
            self.register_hook(SaveFeatureHook(), None, "LOWEST")
        if self.args.use_incl:
            self.register_hook(INCLHook(), None, "LOWEST")
        if self.args.use_wandb:
            self.register_hook(WANDBHook(), None, "LOWEST")
        if self.args.use_aim:
            self.register_hook(AimHook(), None, "LOWEST")

    def process_batch(self, input_args=None, **kwargs):
        """
        process batch data, send data to cuda
        NOTE **kwargs should have the same arguments to train_step function as keys to work properly
        """
        if input_args is None:
            input_args = signature(self.train_step).parameters # 
            input_args = list(input_args.keys())

        input_dict = {}

        for arg, var in kwargs.items():
            if not arg in input_args:
                # discard the key, named idx_lb or idx_ulb, in kwargs
                continue 
            
            if var is None:
                continue
            
            # send var to cuda
            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var
        return input_dict
    

    def process_out_dict(self, out_dict=None, **kwargs):
        """
        process the out_dict as return of train_step
        """
        if out_dict is None:
            out_dict = {}

        for arg, var in kwargs.items():
            out_dict[arg] = var
        
        # process res_dict, add output from res_dict to out_dict if necessary
        return out_dict


    def process_log_dict(self, log_dict=None, prefix='train', **kwargs):
        """
        process the tb_dict as return of train_step
        """
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f'{prefix}/' + arg] = var
        return log_dict

    def compute_prob(self, logits):
        return torch.softmax(logits, dim=-1)

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        """
        train_step specific to each algorithm
        """
        # implement train step for each algorithm
        # compute loss
        # update model 
        # record log_dict
        # return log_dict
        raise NotImplementedError


    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            
            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1
            
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")


    def evaluate(self, eval_dest='eval', out_key='logits'):
        """
        evaluation function
        """
        self.model.eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true, y_pred, indices = [], [], []
        with torch.no_grad():
            for data in eval_loader:
                x = data['x_lb']
                y = data['y_lb']
                idxs = data['idx_lb']
                
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)
                idxs = idxs.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                logits = self.model(x)[out_key]
                
                loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)

                indices.append(idxs)
                y_true.append(y)
                y_pred.append(torch.max(logits, dim=-1)[1])
                total_loss += loss.item() * num_batch
            
            indices = torch.cat(indices)
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)

            if self.distributed and self.world_size > 1:
                indices = concat_all_gather(indices)
                y_true = concat_all_gather(y_true)
                y_pred = concat_all_gather(y_pred)

                ranks = len(eval_loader.dataset) % dist.get_world_size()
                indices = remove_wrap_arounds(indices, ranks)
                y_true = remove_wrap_arounds(y_true, ranks)
                y_pred = remove_wrap_arounds(y_pred, ranks)

                sorted_indices = torch.argsort(indices)
                indices = indices[sorted_indices]
                y_true = y_true[sorted_indices]
                y_pred = y_pred[sorted_indices]

            assert torch.all(indices == torch.arange(len(eval_loader.dataset)).cuda(self.gpu))
                
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        top1 = accuracy_score(y_true, y_pred)
        # balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        # precision = precision_score(y_true, y_pred, average='macro')
        # recall = recall_score(y_true, y_pred, average='macro')
        # F1 = f1_score(y_true, y_pred, average='macro')

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        acc_per_class = (cf_mat.diagonal() / cf_mat.sum(axis=1) * 100.0).round(2)
        self.print_fn('confusion matrix:\n' + np.array_str(acc_per_class))
        self.ema.restore()
        self.model.train()

        eval_dict = {eval_dest+'/top-1-acc': top1 
                    # eval_dest+'/loss': total_loss / total_num, 
                    #  eval_dest+'/balanced_acc': balanced_top1, 
                    #  eval_dest+'/precision': precision, 
                    #  eval_dest+'/recall': recall, 
                    #  eval_dest+'/F1': F1,
                    }
        return eval_dict


    def get_save_dict(self):
        """
        make easier for saving model when need save additional arguments
        """
        # base arguments for all models
        save_dict = {
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_scaler': self.loss_scaler.state_dict(),
            'it': self.it,
            'epoch': self.epoch,
            'best_it': self.best_it,
            'best_eval_acc': self.best_eval_acc,
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        return save_dict
    

    def save_model(self, save_name, save_path):
        """
        save model and specified parameters for resume
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_filename = os.path.join(save_path, save_name)
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)
        self.print_fn(f"model saved: {save_filename}")


    def load_model(self, load_path):
        """
        load model and specified parameters for resume
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        missing_keys = self.model.load_state_dict(checkpoint['model'])
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        if 'it' in checkpoint:
            self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
            self.it = checkpoint['it']
            self.start_epoch = checkpoint['epoch']
            self.epoch = self.start_epoch
            self.best_it = checkpoint['best_it']
            self.best_eval_acc = checkpoint['best_eval_acc']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler is not None and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.print_fn('Resume training from it: {}, epoch: {}, missing_keys: {}'.format(self.it, self.epoch, missing_keys))
        else:
            self.print_fn('[load_model in algorithmbase.py] Source Model weight loaded, missing_keys: {}'.format(missing_keys))
        return checkpoint
    

    def check_prefix_state_dict(self, state_dict):
        """
        remove prefix state dict in ema model
        """
        new_state_dict = dict()
        for key, item in state_dict.items():
            if key.startswith('module'):
                new_key = '.'.join(key.split('.')[1:])
            else:
                new_key = key
            new_state_dict[new_key] = item
        return new_state_dict

    def register_hook(self, hook, name=None, priority='NORMAL'):
        """
        Ref: https://github.com/open-mmlab/mmcv/blob/a08517790d26f8761910cac47ce8098faac7b627/mmcv/runner/base_runner.py#L263
        Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            hook_name (:str, default to None): Name of the hook to be registered. Default is the hook class name.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        hook.name = name if name is not None else type(hook).__name__

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        
        if not inserted:
            self._hooks.insert(0, hook)

        # call set hooks
        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook


    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            hook_name (str): The specific hook name to be called, such as
                "param_update" or "dist_align", uesed to call single hook in train_step.
        """
        
        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)
        
        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

    def registered_hook(self, hook_name):
        """
        Check if a hook is registered
        """
        return hook_name in self.hooks_dict


    @staticmethod
    def get_argument():
        """
        Get specificed arguments into argparse for each algorithm
        """
        return {}
    

    def redefine_data_for_src(self):
        # set source domain dataset
        src_domain = os.path.basename(self.args.src_model_path).split('_')[0]
        # TODO: hard code here, need to be changed
        num_label_dump = 120 if src_domain == 'visda' else 1260
        
        origin_trg = self.args.trg
        self.args.trg = src_domain
        src_dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, num_label_dump, self.args.num_classes, self.args.data_dir, self.args.include_lb_to_ulb)
        self.args.trg = origin_trg
        self.dataset_dict['eval_src'] = src_dataset_dict['eval']
        del src_dataset_dict

        # set source domain data loader
        self.loader_dict['eval_src'] =  get_data_loader(self.args,
                                                   self.dataset_dict['eval_src'],
                                                   self.args.eval_batch_size,
                                                   # make sure data_sampler is None for evaluation
                                                   data_sampler=None,
                                                   num_workers=self.args.num_workers,
                                                   distributed=self.distributed,
                                                   drop_last=False)


    def redefine_train_lb(self):
        dump_dir = os.path.join(self.args.data_dir, self.args.dataset, 'labeled_idx')

        os.makedirs(dump_dir, exist_ok=True)
        src_domain = os.path.basename(self.args.src_model_path).split('_')[0]
        lb_dump_path = os.path.join(dump_dir, f'lb_labels{self.args.num_labels}_{src_domain[0]}2{self.args.trg[0]}_seed{self.args.seed}_NBF_idx.npy')

        if os.path.exists(lb_dump_path):
            self.print_fn("[RedefineDataHook] load lb_idx by NRF from {}".format(lb_dump_path))
            lb_idx = np.load(lb_dump_path)
        else:
            dataset_dump = deepcopy(self.dataset_dict['train_ulb'])
            dataset_dump.is_ulb = False
            dataloader_dump =  get_data_loader(self.args, dataset_dump, self.args.eval_batch_size, 
                                        data_sampler=None, num_workers=self.args.num_workers, distributed=self.distributed, drop_last=False)
            # divide false and true samples
            self.model.eval()
            self.ema.apply_shadow()
            y_gt = []
            y_pred = []
            index = []
            with torch.no_grad():
                for data in dataloader_dump:
                    x = data['x_lb']
                    y = data['y_lb']
                    idx = data['idx_lb']
                    
                    if isinstance(x, dict):
                        x = {k: v.cuda(self.gpu) for k, v in x.items()}
                    else:
                        x = x.cuda(self.gpu)
                    y = y.cuda(self.gpu)

                    num_batch = y.shape[0]
                    logits = self.model(x)['logits']
                    y_gt.extend(y.cpu().tolist())
                    y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                    index.extend(idx.tolist())

            y_gt = np.array(y_gt)
            y_pred = np.array(y_pred)
            index = np.array(index)
            self.ema.restore()
            self.model.train()
            # get samples
            assert self.args.num_labels != 0, "num_labels should not be 0"
            lb_samples_per_class = [int(self.args.num_labels / self.num_classes)] * self.num_classes
            
            lb_idx = []
            num_false = []
            all_falses = np.where(y_pred != y_gt)[0]
            all_trues = np.where(y_pred == y_gt)[0]
            for c in range(self.num_classes):
                # pick #lb_samples_per_class[c] samples from idx_false
                idx_gt = np.where(y_gt == c)[0]
                idx_false = np.intersect1d(all_falses, idx_gt)
                num_false.append(len(idx_false))
                if len(idx_false) >= lb_samples_per_class[c]:
                    np.random.shuffle(idx_false)
                    idxs = idx_false[:lb_samples_per_class[c]]
                else:
                    # if not enough, pick #lb_samples_per_class[c] - len(idx_false) samples from idx_true
                    idx_true = np.intersect1d(all_trues, idx_gt)
                    np.random.shuffle(idx_true)
                    idxs = np.concatenate([idx_false, idx_true[:lb_samples_per_class[c] - len(idx_false)]])
                lb_idx.extend(idxs)
            self.print_fn("[RedefineDataHook] num false count: {}".format(str(num_false))) 
            lb_idx = np.array(lb_idx)
            np.save(lb_dump_path, lb_idx)

        # check the number of classes in lb_idx
        datas=self.dataset_dict['train_ulb'].data[lb_idx]
        targets=self.dataset_dict['train_ulb'].targets[lb_idx]
        lb_count = [0 for _ in range(self.num_classes)]
        for c in targets:
            lb_count[c] += 1
        
        self.print_fn("[RedefineDataHook] new lb_count: {}".format(str(lb_count)))
        # redefine dataset and dataloader
        self.dataset_dict['train_lb'].data = datas
        self.dataset_dict['train_lb'].targets = targets



class ImbAlgorithmBase(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        
        # imbalanced arguments
        self.lb_imb_ratio = self.args.lb_imb_ratio
        self.ulb_imb_ratio = self.args.ulb_imb_ratio
        self.imb_algorithm = self.args.imb_algorithm
    
    def imb_init(self, *args, **kwargs):
        """
        intiialize imbalanced algorithm parameters
        """
        pass 

    def set_optimizer(self):
        if 'vit' in self.args.net and self.args.dataset in ['cifar100', 'food101', 'semi_aves', 'semi_aves_out']:
            return super().set_optimizer() 
        elif self.args.dataset in ['imagenet', 'imagenet127']:
            return super().set_optimizer() 
        else:
            self.print_fn("Create optimizer and scheduler")
            optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay, bn_wd_skip=False)
            scheduler = None
            return optimizer, scheduler