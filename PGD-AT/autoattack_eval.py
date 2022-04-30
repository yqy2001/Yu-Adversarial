import os
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
# from wideresnet import *
from models import *
# from models.resnet_cifar import ResNet18 as ResNet18_multibn
from models.resnet_cifar_multibn import resnet18 as ResNet18_multibn
from utils.utils import load_BN_checkpoint, load_BN_checkpoint_ce
from functools import partial

import sys
sys.path.insert(0, '..')

# from resnet import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='resnet18', choices=['resnet18', 'resnet18_multibn'])
    parser.add_argument('--data_dir', type=str, default='~/data')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--model', type=str, default='./model_test.pt')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='standard')
    
    args = parser.parse_args()

    # load model
    
    if args.model_type == "resnet18":
        model = ResNet18().cuda()
    elif args.model_type == "resnet18_multibn":
        bn_names = ['normal', 'pgd', 'pgd_ce']
        model = ResNet18_multibn(bn_names=bn_names)
        # model = ResNet18_multibn()
        model = model.cuda()
        
    # elif args.model_type == "WideResNet":
    #     model = WideResNet().cuda()
    
    model = nn.DataParallel(model)
    ckpt = torch.load(args.model)
    if args.model_type == "resnet18_multibn":
        state_dict = ckpt['state_dict']
        # new_state_dict, state_dict_normal = load_BN_checkpoint(state_dict)
        # new_state_dict, state_dict_normal = load_BN_checkpoint_ce(state_dict)
        model_dict = model.state_dict()
        # new_state_dict = load_pretrain_checkpoint2finetune(model_dict, state_dict)
        # new_model_dict = load_supcon_checkpoint2finetune(model_dict, state_dict)
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    # model.load_state_dict(ckpt['state_dict'])

    model.cuda()
    model.eval()
    
    if args.model_type == "resnet18_multibn":
        model = partial(model, bn_name="pgd_ce")

    '''
    model = ResNet18()
    ckpt = torch.load(args.model)
    model.load_state_dict(ckpt)
    model.cuda()
    model.eval()
    '''
    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)
    
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # load attack    
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
        version=args.version)
    
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    
    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2
    
    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                bs=args.batch_size)
            
            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                args.save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon))

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                y_test[:args.n_ex], bs=args.batch_size)
            
            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap.pth'.format(
                args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))
                
