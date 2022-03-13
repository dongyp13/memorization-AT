from __future__ import print_function
from __future__ import division
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from PIL import Image

from models.wideresnet import *
from utils import corrupt_dataset

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-policy', type=str, default='step',
                    choices=['step', 'multi_step', 'cosine'], help='learning rate decay method')
parser.add_argument('--lr-step-size', type=int, default=100,
                    help='step size for lr decay')
parser.add_argument('--lr-milestones', default=[100,150], 
                    help='milestones for lr decay')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=28,
                    help='model depth')
parser.add_argument('--widen-factor', type=int, default=10,
                    help='model width')

parser.add_argument('--epsilon', default=8.0/255.0,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=2.0/255.0,
                    help='perturb step size')
parser.add_argument('--random-start', action='store_true', default=True)
parser.add_argument('--beta', type=float, default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--loss-type', type=str, default='cross_entropy', 
                    choices=['cross_entropy', 'trades'],
                    help='loss type for training')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--data-augmentation', action='store_true', default=False)

parser.add_argument('--noise-type', type=str, default='none', 
                    choices=['none', 'label_symmetric'], help='noise type')
parser.add_argument('--noise-rate', type=float, default=0.0, help='noise rate')
parser.add_argument('--drop-rate', type=float, default=0.0, help='dropout rate')

args = parser.parse_args()
print(args)


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    with torch.no_grad():
        logits = model(x_natural)
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(logits, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(logits, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


def pgd_loss(model,
             x_natural,
             y,
             optimizer,
             step_size=0.003,
             epsilon=0.031,
             perturb_steps=10,
             random_start=True,
             distance='l_inf'):

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    if random_start:
        x_adv = x_natural.detach() + torch.FloatTensor(*x_natural.shape).uniform_(-epsilon, epsilon).cuda()
    else:
        x_adv = x_natural.detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    return loss


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
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
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'svhn':
    trainset = torchvision.datasets.SVHN(root='../data', split='train', download=True, transform=transform_test)
    testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)
else:
    raise NotImplementedError

noisy_trainset = corrupt_dataset(trainset, args.noise_type, args.noise_rate, nb_classes=100 if args.dataset == 'cifar100' else 10, model_dir=args.model_dir)

noisy_train_loader = torch.utils.data.DataLoader(noisy_trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch):
    loss_avg = AverageMeter()
    norm_avg = AverageMeter()
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        if args.loss_type == 'trades':
            loss = trades_loss(model=model,
                               x_natural=data,
                               y=target,
                               optimizer=optimizer,
                               step_size=args.step_size,
                               epsilon=args.epsilon,
                               perturb_steps=args.num_steps,
                               beta=args.beta)

        elif args.loss_type == 'cross_entropy':
            loss = pgd_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps,
                            random_start=args.random_start)

        loss.backward()
        norm = 0
        for name, param in model.named_parameters():
            norm += param.grad.norm(2).item() ** 2
        optimizer.step()
        loss_avg.update(loss.item(), args.batch_size)
        norm_avg.update(norm)

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tNorm: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_avg.avg, norm_avg.avg))
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tNorm: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss_avg.avg, norm_avg.avg))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
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
        for data, target in test_loader:
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


def eval_pgd(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        x_adv = data.detach() + torch.FloatTensor(*data.shape).uniform_(-args.epsilon, args.epsilon).cuda()
        for _ in range(args.num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(model(x_adv), target)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, data - args.epsilon), data + args.epsilon)
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


def main():
    
    model = WideResNet(depth=args.depth, widen_factor=args.widen_factor, num_classes=100 if args.dataset == 'cifar100' else 10, dropRate=args.drop_rate).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)
    elif args.lr_policy == 'multi_step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        raise NotImplementedError

    best_acc = 0
    best_robust_acc = 0

    for epoch in range(1, args.epochs + 1):

        # adversarial training
        train(args, model, device, noisy_train_loader, optimizer, epoch)

        # adjust learning rate for SGD
        scheduler.step()

        print('================================================================')
        eval_train(model, device, noisy_train_loader)
        eval_pgd(model, device, noisy_train_loader)
        eval_test(model, device, test_loader)
        eval_pgd(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
