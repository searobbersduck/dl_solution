import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models

import torch.cuda
import torch.utils.model_zoo as model_zoo

import models
import utils

import ml_metrics

model_names = models.resnet_ft_model_names


parser = argparse.ArgumentParser(description='PyTorch Diabetic Retinopathy Training')
parser.add_argument('data', metavar='DIR',
                    help='path to diabetic retinopathy dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
parser.add_argument('--pretrained', dest='pretrained', default=None, help='use pre-trained model')
parser.add_argument('-best-model', dest='best_model', default='check_point', metavar='best model',
                    help='output best model')
parser.add_argument('-optimizer', '--optimizer', default=2, type=int, metavar='N',
                    help='0:SGD,1:Adam,2:Adadelta,3:Adagrad,4:RMSprop')
# run on which gpu device/ -l 0 1 2 3
parser.add_argument('-dev','--devlist', nargs='+', help='<Required> Set flag',
                    type=int, default=[0,1,2,3], required=False)
# train on which image size
parser.add_argument('--image_size', '-img-size', type=int, default=512,
                    choices=utils.IMAGE_SIZE,
                    help='image size: ' +
                        ' | '.join(str(utils.IMAGE_SIZE)) +
                        ' (default: 512)')


import torch.nn as nn
import math

use_cuda = torch.cuda.is_available()

args = parser.parse_args()
devs = args.devlist

if use_cuda:
    torch.cuda.set_device(devs[0])

def main():
    global best_prec1, best_kappa
    best_kappa = 0
    best_prec1 = 0
    args = parser.parse_args()

    image_size = args.image_size

    rescale_size = 512
    crop_size = 448
    downsample = False
    if image_size == 256 | image_size == 128:
        rescale_size = 256
        crop_size = 224
        downsample = True

    fine_tune_label = False

    # create model
    if args.pretrained is not None:
        print("=>creating model 'resnet-{}'".format(args.arch))
        print("fine tune")
        fine_tune_label = True
        model = models.ResNet_FT(args.arch, True, args.pretrained, downsample=downsample)
    else:
        print("=>creating model 'resnet-{}'".format(args.arch))
        model = models.ResNet_FT(args.arch, downsample=downsample)

    if use_cuda:
        model = torch.nn.DataParallel(model, device_ids=devs).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            best_kappa = checkpoint['best_kappa']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if use_cuda:
        cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    traindir = 'sample'
    valdir = 'sample'
    normalize = transforms.Normalize(mean=utils.IMAGENET_MEAN,
                                     std=utils.IMAGENET_STD)

    train_loader = torch.utils.data.DataLoader(
        utils.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        utils.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(rescale_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # define loss function (criterion) and optimizer
    criterion = None
    if use_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    paramslist = None

    if use_cuda:
        if fine_tune_label:
            paramslist = [{'params': model.module.fc.parameters()},
                          ]
        else:
            paramslist = [{'params': model.parameters()}]
    else:
        if fine_tune_label:
            paramslist = [{'params': model.fc.parameters()},
                          ]
        else:
            paramslist = [{'params': model.parameters()}]



    optimizer = torch.optim.SGD(paramslist, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    is_adjust_lr = True

    if args.optimizer == 0:
        optimizer = torch.optim.SGD(paramslist, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        is_adjust_lr = True
    elif args.optimizer == 1:
        optimizer = torch.optim.Adam(paramslist)
    elif args.optimizer == 2:
        optimizer = torch.optim.Adadelta(paramslist)
    elif args.optimizer == 3:
        optimizer = torch.optim.Adagrad(paramslist)
    elif args.optimizer == 4:
        optimizer = torch.optim.RMSprop(paramslist)

    else:
        is_adjust_lr = True

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if is_adjust_lr:
            adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1,kappa = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        is_best = kappa > best_kappa
        best_prec1 = max(prec1, best_prec1)
        best_kappa = max(kappa, best_kappa)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_kappa': best_kappa
        }, is_best, args.best_model)


def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    losses = AverageMeter()
    prec = AverageMeter()
    model.train()

    reslist = []
    targetlist = []

    nProcessed = 0
    nTrain = len(train_loader.dataset)
    for i, (input, target, path) in enumerate(train_loader):
        for t in target:
            targetlist.append(t)
        if use_cuda:
            input, target = input.cuda(), target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        losses.update(loss.data[0], input.size(0))

        pred = output.data.max(1)[1]

        outlist=None
        if use_cuda:
            outlist = pred.cpu().numpy()
        else:
            outlist = pred.numpy()

        for o in outlist:
            reslist.append(o[0])

        correct = None
        if use_cuda:
            correct = pred.eq(target).cpu().numpy().sum()
        else:
            correct = pred.eq(target).numpy().sum()

        correct = 100.*correct/len(input)

        prec.update(correct, input.size(0))

        partialEpoch = epoch+i/len(train_loader)-1

        nProcessed += len(input)

        kp = ml_metrics.quadratic_weighted_kappa(targetlist, reslist, 0, 4)

        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {loss.val:.4f} ({loss.avg:.4f})\t'
              'Precious: {prec.val:.3f} ({prec.avg:.3f})\t'
              'Quadratic_weighted_kappa: {kp:.4f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * i / len(train_loader),
            loss=losses, prec=prec, kp=kp))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(val_loader, model, criterion):
    # switch to evaluate mode
    losses = AverageMeter()
    prec = AverageMeter()
    model.eval()

    reslist = []
    targetlist = []

    nProcessed = 0
    nTrain = len(val_loader.dataset)

    correct1 = 0

    for i, (input, target, path) in enumerate(val_loader):

        for t in target:
            targetlist.append(t)

        if use_cuda:
            input, target = input.cuda(), target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        losses.update(loss.data[0], input.size(0))

        # measure accuracy and record loss
        pred = output.data.max(1)[1]

        outlist = None
        if use_cuda:
            outlist = pred.cpu().numpy()
        else:
            outlist = pred.numpy()

        for o in outlist:
            reslist.append(o[0])

        correct = None
        if use_cuda:
            correct = pred.eq(target).cpu().numpy().sum()
        else:
            correct = pred.eq(target).numpy().sum()

        correct1 += correct
        correct = 100.*correct/len(input)

        prec.update(correct, input.size(0))

        nProcessed += len(input)

        # kp = ml_metrics.quadratic_weighted_kappa(targetlist, reslist, 0, 4)



    kp = ml_metrics.quadratic_weighted_kappa(targetlist, reslist, 0, 4)
    print('quadratic weighted kappa: {}'.format(kp))

    nTotal = len(val_loader.dataset)
    cor = 100.*correct1/nTotal
    print('\nTest set: Average loss: {:.4f}, Precious: {}/{} ({:.0f}%)\t'
          'Quadratic_weighted_kappa: {:.4f}\n'.format(
        losses.avg, correct1, nTotal, cor, kp))

    return cor,kp


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_model_{}'.format(filename))


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 40))
    lr = max(lr,0.00003)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
