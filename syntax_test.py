# #!/usr/bin/env python3
#
# import argparse
# import os
# import numpy as np
#
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# plt.style.use('bmh')
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('expDir', type=str)
#     args = parser.parse_args()
#
#     # trainP = os.path.join(args.expDir, 'train.csv')
#     # trainData = np.loadtxt(trainP, delimiter=',').reshape(-1, 3)
#     # testP = os.path.join(args.expDir, 'test.csv')
#     # testData = np.loadtxt(testP, delimiter=',').reshape(-1, 3)
#     #
#     # N = 392*2 # Rolling loss over the past epoch.
#     #
#     # trainI, trainLoss, trainErr = np.split(trainData, [1,2], axis=1)
#     # trainI, trainLoss, trainErr = [x.ravel() for x in
#     #                                (trainI, trainLoss, trainErr)]
#     # trainI_, trainLoss_, trainErr_ = rolling(N, trainI, trainLoss, trainErr)
#     #
#     # testI, testLoss, testErr = np.split(testData, [1,2], axis=1)
#
#     fig, ax = plt.subplots(1, 1, figsize=(6, 5))
#     # plt.plot(trainI, trainLoss, label='Train')
#     # plt.plot(trainI_, trainLoss_, label='Train')
#     # plt.plot(testI, testLoss, label='Test')
#     plt.xlabel('Epoch')
#     plt.ylabel('Cross-Entropy Loss')
#     plt.legend()
#     ax.set_yscale('log')
#     loss_fname = os.path.join(args.expDir, 'loss.png')
#     plt.savefig(loss_fname)
#     print('Created {}'.format(loss_fname))
#
#     fig, ax = plt.subplots(1, 1, figsize=(6, 5))
#     # plt.plot(trainI, trainErr, label='Train')
#     # plt.plot(trainI_, trainErr_, label='Train')
#     # plt.plot(testI, testErr, label='Test')
#     plt.xlabel('Epoch')
#     plt.ylabel('Error')
#     ax.set_yscale('log')
#     plt.legend()
#     err_fname = os.path.join(args.expDir, 'error.png')
#     plt.savefig(err_fname)
#     print('Created {}'.format(err_fname))
#
#     loss_err_fname = os.path.join(args.expDir, 'loss-error.png')
#     os.system('convert +append {} {} {}'.format(loss_fname, err_fname, loss_err_fname))
#     print('Created {}'.format(loss_err_fname))
#
# def rolling(N, i, loss, err):
#     i_ = i[N-1:]
#     K = np.full(N, 1./N)
#     loss_ = np.convolve(loss, K, 'valid')
#     err_ = np.convolve(err, K, 'valid')
#     return i_, loss_, err_
#
# if __name__ == '__main__':
#     main()


# import argparse
#
# import scipy.misc
#
# import numpy as np
#
# import preprocessing
#
# from PIL import Image
#
# import skimage
# from skimage.filters import threshold_otsu
#
# parser = argparse.ArgumentParser()
#
# parser.add_argument('-image', help = 'the image to preprocessed')
#
# args = parser.parse_args()
#
# image = args.image
#
# img = scipy.misc.imread(image)
#
# img = img.astype(np.float32)
# img /= 255
#
# img_ahe = preprocessing.channelwise_ahe(img)
#
# pilImage = Image.fromarray(skimage.util.img_as_ubyte(img_ahe))
#
# pilImage.show()


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
import torchvision.models as models

import torch.utils.model_zoo as model_zoo


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
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
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

best_prec1 = 0


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


import torch.nn as nn
import math


class ResNet1(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion * 4, 512)
        self.fc1 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc1(x)

        return x

class ResNet2(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet2,self).__init__()
        self.pretrained = pretrained
        self.conv1 = conv3x3(3, 32, stride=2)
        self.bn1 = nn.BatchNorm2d(32)

        m = self.conv1
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))

        m = self.bn1
        m.weight.data.fill_(1)
        m.bias.data.zero_()

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(32, 3)
        self.bn2 = nn.BatchNorm2d(3)

        m = self.conv2
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))

        m = self.bn2
        m.weight.data.fill_(1)
        m.bias.data.zero_()


        self.model = models.ResNet(Bottleneck, [3, 8, 36, 3],num_classes=1000)
        # self.model.load_state_dict(model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth'))

        # self.model = models.ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        self.model.load_state_dict(torch.load('1.pth'))

        for param in self.model.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(1000,5)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.model(x)
        x = self.fc(x)

        return x



def resnet181(pretrained=False):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet1(BasicBlock, [2, 2, 2, 2],num_classes=5)
    if pretrained:
        print('hello resnet18')
    return model



def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # model = resnet181()
    model = ResNet2()

    # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #     model.features = torch.nn.DataParallel(model.features)
    #     model.cuda()
    # else:
    #     model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    traindir='sample'
    valdir='sample'
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(traindir, transforms.Compose([
    #         transforms.RandomSizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),batch_size=4, shuffle=True)


    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(512),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ])),batch_size=4, shuffle=False)


    # define loss function (criterion) and pptimizer
    # criterion = nn.CrossEntropyLoss().cuda()

    class_weight = torch.FloatTensor([1.3609453700116234, 14.378223495702006,
                                      6.637566137566138, 40.235967926689575,
                                      49.612961299435028248494350282484])

    criterion = nn.CrossEntropyLoss(weight=class_weight)

    optimizer = torch.optim.SGD([{'params':model.conv1.parameters(), 'lr':5e-1},
                                 {'params':model.conv2.parameters(), 'lr':5e-1},
                                 {'params':model.bn1.parameters(), 'lr':5e-1},
                                 {'params':model.bn2.parameters(), 'lr':5e-1},
                                 {'params':model.relu.parameters(), 'lr':5e-1},
                                 {'params':model.fc.parameters(), 'lr':5e-1},
                                 ],
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    cnt = 0
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        # print(input)
        # print(target)
        cnt = cnt+1
        data_time.update(time.time() - end)

        # target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        # print(output)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    # print('cnt::::::::::::::::::::::::{}'.format(cnt))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        # print('input size: {}'.format(input.size(0)))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # print(output.data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    print('top1 count: {}'.format(top1.count))
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # print('output:')
    # print(output)
    # print('pred:')
    # print(pred)
    pred = pred.t()
    # print('target:')
    # print(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print('target view expand: ')
    # print(target.view(1,-1).expand_as(pred))
    # print('correct')
    # print(correct)

    res = []
    for k in topk:
        print
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
