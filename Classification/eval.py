import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from semilearn.core.utils import get_net_builder, get_dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


config_data = {
    'visda': 
        {'img_size': 256,
         'crop_ratio': 0.875,
         'num_classes': 12,
         'net': 'resnet101',
         'net_from_torch': True,
         },
    'visda_vit': 
        {'dataset': 'visda',
         'img_size': 256,
         'crop_ratio': 0.875,
         'num_classes': 12,
         'net': 'vit_small_patch16_224',
         'net_from_torch': False,
         },
    'domainnet': 
        {'img_size': 256,
         'crop_ratio': 0.875,
         'num_classes': 126,
         'net': 'resnet50',
         'net_from_torch': True,
         },
    'domainnet_vit': 
        {'dataset': 'domainnet',
         'img_size': 256,
         'crop_ratio': 0.875,
         'num_classes': 126,
         'net': 'vit_small_patch16_224',
         'net_from_torch': False,
         },
    'officehome': 
        {'img_size': 256,
         'crop_ratio': 0.875,
         'num_classes': 65,
         'net': 'resnet50', # resnet50
         'net_from_torch': True,
        #  'trg_eval_mode': 'finetune!=valset',
         },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_model_path', type=str, required=True)
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='wrn_28_2')
    parser.add_argument('--net_from_torch', type=bool, default=True)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--src', type=str, default=None)
    parser.add_argument('--trg', type=str, default=None)
    parser.add_argument('--trg_eval_mode', default='finetune!=valset', type=str, help='options: finetune=valset, finetune!=valset') 
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--partial_ratio', type=int, default=1)             
    parser.add_argument('--crop_ratio', type=int, default=0.875)       
    parser.add_argument('--max_length', type=int, default=512)              # for audio
    parser.add_argument('--max_length_seconds', type=float, default=4.0)    # for audio
    parser.add_argument('--sample_rate', type=int, default=16000)           # for audio

    args = parser.parse_args()

    if args.dataset in config_data:
        config = config_data[args.dataset]
        for key, value in config.items():
            setattr(args, key, value)
    
    checkpoint_path = os.path.join(args.src_model_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['ema_model']
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    args.save_dir = save_dir
    args.save_name = ''
    
    net = get_net_builder(args.net, args.net_from_torch)(num_classes=args.num_classes)
    keys = net.load_state_dict(load_state_dict)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    # specify these arguments manually 
    args.num_labels = args.num_classes
    args.lb_imb_ratio = 1
    args.ulb_imb_ratio = 1
    args.seed = 0
    args.epoch = 1
    args.ulb_num_labels = None
    # args.num_train_iter = 1024
    dataset_dict = get_dataset(args, 'fixmatch', args.dataset, args.num_labels, args.num_classes, args.data_dir, False)
    eval_dset = dataset_dict['eval']
    eval_loader = DataLoader(eval_dset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4)
    print("evaluation batch size: {}".format(args.batch_size))
 
    acc = 0.0
    test_feats = []
    test_preds = []
    test_probs = []
    test_labels = []
    with torch.no_grad():
        for data in tqdm(eval_loader):
            image = data['x_lb']
            target = data['y_lb']

            image = image.type(torch.FloatTensor).cuda()
            feat = net(image, only_feat=True)
            logit = net(feat, only_fc=True)
            prob = logit.softmax(dim=-1)
            pred = prob.argmax(1)

            acc += pred.cpu().eq(target).numpy().sum()

           # feature save
            # test_feats.append(feat.cpu().numpy())
            # test_probs.append(prob.cpu().numpy())
            test_preds.append(pred.cpu().numpy())
            test_labels.append(target.cpu().numpy())
    # test_feats = np.concatenate(test_feats)
    # test_probs = np.concatenate(test_probs)
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)

    print(f"Test Accuracy: {acc/len(eval_dset)}")
    
    cf_mat = confusion_matrix(test_labels, test_preds, normalize='true')
    acc_per_class = (cf_mat.diagonal() / cf_mat.sum(axis=1) * 100.0).round(2)
    print('confusion matrix:\n' + np.array_str(acc_per_class))

    # top1 = accuracy_score(test_labels, test_preds)
    # balanced_top1 = balanced_accuracy_score(test_labels, test_preds)
    # precision = precision_score(test_labels, test_preds, average='macro')
    # recall = recall_score(test_labels, test_preds, average='macro')
    # F1 = f1_score(test_labels, test_preds, average='macro')