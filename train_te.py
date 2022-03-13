from __future__ import print_function
from __future__ import division
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

from models.wideresnet import *
from models.resnet import *
from models.vgg import *
from loss import PGD_TE, TRADES_TE
from dataset import CIFAR10, CIFAR100, SVHN

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-policy', type=str, default='step',
                    choices=['step', 'cosine'], help='learning rate decay method')
parser.add_argument('--lr-milestones', default=[100,150], 
                    help='milestones for lr decay')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=34,
                    help='model depth')
parser.add_argument('--widen-factor', type=int, default=10,
                    help='model width')
parser.add_argument('--model', type=str, default='ResNet18')

parser.add_argument('--epsilon', default=8.0/255.0,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=2.0/255.0,
                    help='perturb step size')
parser.add_argument('--norm', type=str, default='linf', help='norm')
parser.add_argument('--beta', type=float, default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=50, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--loss-type', type=str, default='cross_entropy', 
                    choices=['cross_entropy', 'trades'],
                    help='loss type for training')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--data-augmentation', action='store_true', default=True)

parser.add_argument('--te-alpha', default=0.9, type=float,
                    help='momentum term of self-adaptive training')
parser.add_argument('--start-es', default=90, type=int,
                    help='start epoch of self-adaptive training (default 0)')
parser.add_argument('--end-es', default=150, type=int,
                    help='start epoch of self-adaptive training (default 0)')
parser.add_argument('--reg-weight', default=300, type=float)

args = parser.parse_args()
print(args)

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# setup data loader
if args.data_augmentation:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == 'cifar10':
    trainset = CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = CIFAR10(root='../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    trainset = CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    testset = CIFAR100(root='../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'svhn':
    trainset = SVHN(root='../data', split='train', download=True, transform=transform_test)
    testset = SVHN(root='../data', split='test', download=True, transform=transform_test)
else:
    raise NotImplementedError


train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(args, model, device, train_loader, criterion, optimizer, epoch, weight):
    model.train()
    print('Train Epoch: {}\treg_weight: {:.4f}'.format(epoch, weight))
    for batch_idx, (data, target, index) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        loss = criterion(data, target, index, epoch, model, optimizer, weight)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def eval_test_pgd(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target, _ in test_loader:
        data, target = data.to(device), target.to(device)
        if args.norm == 'linf':
            x_adv = data.detach() + torch.FloatTensor(*data.shape).uniform_(-args.epsilon, args.epsilon).cuda()
        elif args.norm == 'l2':
            x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
        for _ in range(args.num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(model(x_adv), target)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            if args.norm == 'linf':
                x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, data - args.epsilon), data + args.epsilon)
            elif args.norm == 'l2':
                g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1,1,1,1)
                scaled_grad = grad.detach() / (g_norm.detach() + 1e-10)
                x_adv = data + (x_adv.detach() + args.step_size * scaled_grad - data).view(data.size(0), -1).renorm(p=2, dim=0, maxnorm=args.epsilon).view_as(data)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        with torch.no_grad():
            output = model(x_adv)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test PGD: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def sigmoid_rampup(current, start_es, end_es):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if current < start_es:
        return 0.0
    if current > end_es:
        return 1.0
    else:
        import math
        phase = 1.0 - (current - start_es) / (end_es - start_es)
        return math.exp(-5.0 * phase * phase)


def main():
    # init model, ResNet18() can be also used here for training
    if args.model == 'WideResNet':
        model = WideResNet(depth=args.depth, widen_factor=args.widen_factor, num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
    elif args.model == 'ResNet18':
        model = ResNet18(num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
    elif args.model == 'VGG16':
        model = VGG('VGG16').to(device)
    
    
    if args.loss_type == 'cross_entropy':
        criterion = PGD_TE(num_samples=len(trainset),
                           num_classes=100 if args.dataset == 'cifar100' else 10,
                           momentum=args.te_alpha,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           norm=args.norm,
                           es=args.start_es)
    elif args.loss_type == 'trades':
        criterion = TRADES_TE(num_samples=len(trainset),
                              num_classes=100 if args.dataset == 'cifar100' else 10,
                              momentum=args.te_alpha,
                              step_size=args.step_size,
                              epsilon=args.epsilon,
                              perturb_steps=args.num_steps,
                              norm=args.norm,
                              es=args.start_es,
                              beta=args.beta)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.lr_policy == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        raise NotImplementedError

    best_acc = 0
    best_robust_acc = 0

    for epoch in range(1, args.epochs + 1):

        rampup_rate = sigmoid_rampup(epoch, args.start_es, args.end_es)
        weight = rampup_rate * args.reg_weight

        # adversarial training
        train(args, model, device, train_loader, criterion, optimizer, epoch, weight)

        # adjust learning rate for SGD
        scheduler.step()

        # evaluation on natural examples
        print('================================================================')
        _, acc = eval_test(model, device, test_loader)
        _, robust_acc = eval_test_pgd(model, device, test_loader)
        print('================================================================')

        save_best = False
        if robust_acc > best_robust_acc:
            best_acc = acc
            best_robust_acc = robust_acc
            save_best = True
        elif robust_acc == best_robust_acc and acc > best_acc:
            best_acc = acc
            best_robust_acc = robust_acc
            save_best = True

        if save_best:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-best.pt'))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-wideres-checkpoint_best.tar'))

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
