from __future__ import print_function
import os
import argparse
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from models.wideresnet import *
from models.resnet import *
from trades import trades_loss

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--name', type=str, default='TRADES_CIFAR10')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

MODEL_MAP = {"WideResNet": WideResNet, "ResNet18": ResNet18, "ResNet34": ResNet34, "ResNet50": ResNet50}
parser.add_argument('--model', default="ResNet18", choices=MODEL_MAP.keys(), metavar='N',
                    help='save frequency')

args = parser.parse_args()

# settings
args.name += '_' + args.model
model_dir = args.model_dir + args.model
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

writer = SummaryWriter("tblogs/"+args.name)

batch_i = 1


def train(args, model, device, train_loader, optimizer, epoch):
    global batch_i

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        writer.add_scalar('Loss/train_rob', loss, batch_i)
        batch_i += 1


def eval_natural(model, device, data_loader, eval_type):
    assert eval_type in ["train", "test"], "eval_type must be train or test"

    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)
    print('{}: Average loss: {:.2f}, Accuracy: {}/{} ({:.0f}%)'.format(
        eval_type, loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    accuracy = correct / len(data_loader.dataset)
    return loss, accuracy


def eval_adv_whitebox(model, device, data_loader, eval_type):
    """
    evaluate model by white-box attack
    """
    global batch_i
    assert eval_type in ["train", "test"], "eval_type must be train or test"

    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust, rob_loss = _pgd_whitebox(model, X, y)
        natural_err_total += err_natural
        robust_err_total += err_robust

        if eval_type == "test":
            writer.add_scalar('Loss/test_rob', rob_loss, batch_i)
            batch_i += 1

    print('{}: Accuracy: {}/{} ({:.0f}%)'.format(
        eval_type + '-natural', len(data_loader.dataset) - natural_err_total, len(data_loader.dataset),
        100. * (len(data_loader.dataset) - natural_err_total) / len(data_loader.dataset)))
    print('{}: Accuracy: {}/{} ({:.0f}%)'.format(
        eval_type + '-robust', len(data_loader.dataset) - robust_err_total, len(data_loader.dataset),
        100. * (len(data_loader.dataset) - robust_err_total) / len(data_loader.dataset)))
    natural_acc = 100. * (len(data_loader.dataset) - natural_err_total) / len(data_loader.dataset)
    robust_acc = 100. * (len(data_loader.dataset) - robust_err_total) / len(data_loader.dataset)

    return natural_acc, robust_acc


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=0.003):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    # random restart
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X_pgd), y)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd, loss


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    global batch_i
    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='/data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='/data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # init model, ResNet18() can be also used here for training
    model = MODEL_MAP[args.model]().to(device)
    # model = WideResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train_nat_acc_total, train_rob_acc_total, test_nat_acc_total, test_rob_acc_total = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)
        batch_i = 1
        # evaluation on natural & robust examples
        print('================================================================')
        # train_natual_acc, train_rob_acc = eval_natural(model, device, train_loader, "train")
        # test_natual_acc, test_rob_acc = eval_natural(model, device, test_loader, "test")
        train_natual_acc, train_rob_acc = eval_adv_whitebox(model, device, train_loader, "train")
        test_natual_acc, test_rob_acc = eval_adv_whitebox(model, device, test_loader, "test")

        writer.add_scalar('Accuracy/train_nat', train_natual_acc, epoch)
        writer.add_scalar('Accuracy/train_rob', train_rob_acc, epoch)
        writer.add_scalar('Accuracy/test_nat', test_natual_acc, epoch)
        writer.add_scalar('Accuracy/test_rob', test_rob_acc, epoch)
        # train_nat_acc_total.append(train_natual_acc)
        # train_rob_acc_total.append(train_rob_acc)
        # test_nat_acc_total.append(test_natual_acc)
        # test_rob_acc_total.append(test_rob_acc)
        end = time.time()
        print("Time cost: {:.0f} s".format(end - start))
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-{}-epoch{}.pt'.format(args.model, epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-{}-checkpoint_epoch{}.tar'.format(args.model, epoch)))

    acc = [train_nat_acc_total, train_rob_acc_total, test_nat_acc_total, test_rob_acc_total]
    with open(args.name + '_acc', 'wb') as f:
        pickle.dump(acc, f)

    # read
    # with open(args.name + '_acc', 'rb') as f:
    #     a = pickle.load(f)


if __name__ == '__main__':
    main()
