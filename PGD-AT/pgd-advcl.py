import os
import argparse
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.backends.cudnn as cudnn

import apex
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST

from models import *
from models.resnet_cifar_multibn import resnet18 as ResNet18_multibn

from losses import SupConLoss

import numpy as np
import attack_generator as attack
from utils import Logger
from utils.utils import TwoCropTransformAdv, progress_bar, adjust_learning_rate

parser = argparse.ArgumentParser(description='PGD-AT')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')

# optim settings
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'onedrop', 'multipledecay', 'cosine'])
parser.add_argument('--lr-max', default=0.1, type=float)
parser.add_argument('--lr-one-drop', default=0.01, type=float)
parser.add_argument('--lr-drop-epoch', default=100, type=int)

parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                    help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')

# adversarial settings
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step-size', type=float, default=0.007, help='step size')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--random',type=bool,default=True,help="whether to initiat adversarial sample with random noise")
parser.add_argument('--only_x_ce', action='store_true',
                    help='only generate x_ce(without x_cl) to compute ce&cl loss')

# model
parser.add_argument('--model', type=str, default="WRN",help="decide which network to use,choose from resnet18,WRN")
parser.add_argument('--multibn', action='store_true', default=False)
parser.add_argument('--depth',type=int,default=32,help='WRN depth')
parser.add_argument('--width-factor',type=int,default=10,help='WRN width factor')
parser.add_argument('--drop-rate',type=float,default=0.0, help='WRN drop rate')

# dataset
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn,cifar100,mnist")

parser.add_argument('--resume',type=str,default=None,help='whether to resume training')
parser.add_argument('--out-dir',type=str,default='./GAIRAT_result',help='dir of output')

# advcl settings
parser.add_argument('--advcl',action='store_true', default=False,help='whether to use advcl as regulation')
parser.add_argument('--advcl_weight', default=1.0, type=float)
parser.add_argument('--stpg',action='store_true', default=False, help='stpg clean in cl loss')
parser.add_argument('--supcl_clean',action='store_true', default=False, help='use supcon in contrast clean pairs')
parser.add_argument('--supcl_adv',action='store_true', default=False, help='use supcon in contrast clean&adv pairs')

# acc_aware
parser.add_argument('--acc_aware', action='store_true',
                    help='using adv probablity to reweight loss')
parser.add_argument('--beta_cl', type=float, default=1.0,
                        help='beta in pow() of p(f(x_adv)==y)')
parser.add_argument('--lambd_cl', type=float, default=1.0,
                    help='coefficient in cl')
parser.add_argument('--temp_cl', type=float, default=10.0,
                    help='temp to calculate alpha')

args = parser.parse_args()

# Training settings
depth = args.depth
width_factor = args.width_factor
drop_rate = args.drop_rate
resume = args.resume
out_dir = args.out_dir

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


# Learning schedules
if args.lr_schedule == 'superconverge':
    lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
elif args.lr_schedule == 'piecewise':
    def lr_schedule(t):
        if args.epochs >= 110:
            # Train Wide-ResNet
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            elif t / args.epochs < (11/12):
                return args.lr_max / 100.
            else:
                return args.lr_max / 200.
        else:
            # Train ResNet
            if t / args.epochs < 0.3:
                return args.lr_max
            elif t / args.epochs < 0.6:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
elif args.lr_schedule == 'linear':
    lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
elif args.lr_schedule == 'onedrop':
    def lr_schedule(t):
        if t < args.lr_drop_epoch:
            return args.lr_max
        else:
            return args.lr_one_drop
elif args.lr_schedule == 'multipledecay':
    def lr_schedule(t):
        return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
elif args.lr_schedule == 'cosine': 
    def lr_schedule(t): 
        return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))

# Store path
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Save checkpoint
def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# Get adversarially robust network
def train(epoch, model, train_loader, optimizer):
    
    lr = 0
    num_data = 0
    train_robust_loss = 0
    model.train()

    for batch_idx, (data_all, target) in enumerate(train_loader):
        loss = 0
        data1, data2, data = data_all
        data1, data2, data, target = data1.cuda(non_blocking=True), data2.cuda(non_blocking=True), data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        
        # Get adversarial data
        if args.advcl and not args.only_x_ce:
            if args.multibn:
                x_adv, x_adv_cl = attack.advcl_PGD(model,(data1, data2, data), target, args.epsilon,args.step_size,args.num_steps,loss_fn="cent",category="Madry",rand_init=True, args=args )
            else:
                x_adv, x_adv_cl = attack.advcl_PGD_singlebn(model,(data1, data2, data), target, args.epsilon,args.step_size,args.num_steps,loss_fn="cent",category="Madry",rand_init=True, args=args ) 
        else:
            x_adv = attack.GA_PGD(model,data,target,args.epsilon,args.step_size,args.num_steps,loss_fn="cent",category="Madry",rand_init=True)

        # model.train()
        # if args.advcl:
        #     adjust_learning_rate(args, optimizer, epoch + 1)
        # else:
        #     lr = lr_schedule(epoch + 1)
        #     optimizer.param_groups[0].update(lr=lr)
        #     optimizer.zero_grad()

        lr = lr_schedule(epoch + 1)
        optimizer.param_groups[0].update(lr=lr)
        optimizer.zero_grad()
        
        logit = model(x_adv)

        p_adv_y = None
        if args.acc_aware: 
            # ======== compute p(model(x_ce)==y) ========
            p_adv_y = torch.ones(len(data)).cuda()
            lgt_sfm = F.softmax(logit/args.temp_cl, dim=1)
            for pp in range(len(logit)):
                L = lgt_sfm[pp][target[pp].item()]
                L = L.pow(args.beta_cl).detach()
                p_adv_y[pp] = L
            
            ce_loss = nn.CrossEntropyLoss(reduce="none")(logit, target)
            ce_loss = ce_loss.mul(1 - p_adv_y)
            ce_loss = ce_loss.mean()
            loss = ce_loss
        else:
            loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)
        
        if args.advcl:
            f1_proj, f1_logits = model(data1, contrast=True)
            f2_proj, f2_logits = model(data2, contrast=True)
            if args.only_x_ce:
                fcl_proj, fcl_logits = model(x_adv, contrast=True)
            else:
                fcl_proj, fcl_logits = model(x_adv_cl, contrast=True)
            features = torch.cat([fcl_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1)], dim=1)

            criterion_cl = SupConLoss(temperature=0.5, args=args)
            cl_loss = criterion_cl(features, labels=target, alpha=p_adv_y)

            loss = loss + args.advcl_weight*cl_loss

        # print('loss:', loss)
        train_robust_loss += loss.item() * len(x_adv)
        
        loss.backward()
        optimizer.step()
        
        num_data += len(data)

        progress_bar(batch_idx, len(train_loader),
                     'Loss: %.3f lr: %.2f (%d)'
                     % (train_robust_loss / ((batch_idx + 1)*args.batch_size), lr, num_data))

    train_robust_loss = train_robust_loss / num_data

    return train_robust_loss, lr


# Setup data loader
transform_strong = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_train_two = TwoCropTransformAdv(transform_strong, transform_train)

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == "cifar10":
    trainset = CIFAR10(root='~/data/', train=True, download=True, transform=transform_train)
    testset = CIFAR10(root='~/data/', train=False, download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    trainset = CIFAR100(root='~/data', train=True, download=True, transform=transform_train)
    testset = CIFAR100(root='~/data', train=False, download=True, transform=transform_test)
elif args.dataset == "svhn":
    trainset = SVHN(root='~/data/', split='train', download=True, transform=transform_train)
    testset = SVHN(root='~/data/', split='test', download=True, transform=transform_test)
elif args.dataset == "mnist":
    trainset = MNIST(root='~/data/', train=True, download=True, transform=transforms.ToTensor())
    testset = MNIST(root='~/data/', train=False, download=True, transform=transforms.ToTensor())
else:
    raise NotImplementedError

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)


def main():
    # Models and optimizer
    if args.model == "resnet18":
        if args.multibn:
            bn_names = ['normal', 'pgd', 'pgd_ce']
            model = ResNet18_multibn(bn_names=bn_names)
        else:
            model = ResNet18()
    elif args.model == "preactresnet18":
        model = PreActResNet18()
    elif args.model == "WRN":
        model = Wide_ResNet_Madry(depth=depth, num_classes=10, widen_factor=width_factor, dropRate=drop_rate)

    print("=====> Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.lr_policy == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        raise NotImplementedError

    # Resume
    title = 'PGD-AT'
    best_acc = 0
    start_epoch = 0
    if resume:
        # Resume directly point to checkpoint.pth.tar
        print ('==> PGD-AT Resuming from checkpoint ..')
        print(resume)
        assert os.path.isfile(resume)
        out_dir = os.path.dirname(resume)
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['test_pgd20_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title, resume=True)
    else:
        print('==> PGD-AT')
        logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title)
        logger_test.set_names(['Epoch', 'Natural Test Acc', 'PGD20 Acc'])

    ## Training get started
    test_nat_acc = 0
    test_pgd20_acc = 0
    best_nat = [0, 0]
    best_pgd20 = [0, 0]

    for epoch in range(start_epoch, args.epochs):

        # Adversarial training
        train_robust_loss, lr = train(epoch, model, train_loader, optimizer)

        # Evalutions similar to DAT.
        _, test_nat_acc = attack.eval_clean(model, test_loader)
        _, test_pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", category="Madry", random=True)

        if test_nat_acc >= best_nat[0]:
            best_nat[0] = test_nat_acc
            best_nat[1] = test_pgd20_acc

        if test_pgd20_acc >= best_pgd20[1]:
            best_pgd20[0] = test_nat_acc
            best_pgd20[1] = test_pgd20_acc

        print(
            'Epoch: [%d | %d] | Learning Rate: %f | Natural Test Acc %.2f | PGD20 Test Acc %.2f |\n' % (
            epoch,
            args.epochs,
            lr,
            test_nat_acc,
            test_pgd20_acc)
            )

        logger_test.append([epoch + 1, test_nat_acc, test_pgd20_acc])

        # Save the best checkpoint
        if test_pgd20_acc > best_acc:
            best_acc = test_pgd20_acc
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'test_nat_acc': test_nat_acc,
                    'test_pgd20_acc': test_pgd20_acc,
                    'optimizer' : optimizer.state_dict(),
                },filename='bestpoint.pth.tar')

        # Save the last checkpoint
        save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'test_nat_acc': test_nat_acc,
                    'test_pgd20_acc': test_pgd20_acc,
                    'optimizer' : optimizer.state_dict(),
                })

    logger_test.write("best nat acc pairs", [best_nat[0], best_nat[1]])
    logger_test.write("best pgd20 acc pairs", [best_pgd20[0], best_pgd20[1]])
    logger_test.close()