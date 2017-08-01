# invoke script: python test_dr.py ./pretrained_models/best_model_test --arch 18 -dev 0

import models

import argparse

from PIL import Image

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


import utils


model_names = models.resnet_ft_model_names

parser = argparse.ArgumentParser()
parser.add_argument('--arch', '-a', metavar='ARCH', default='18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('model', metavar='dr model', help = 'the trained model')
# run on which gpu device/ -l 0 1 2 3
parser.add_argument('-dev','--devlist', nargs='+', help='<Required> Set flag',
                    type=int, default=[0,1,2,3], required=False)

args = parser.parse_args()
devs = args.devlist

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.cuda.set_device(devs[0])

# 目前只支持train from scatch的模型
class DrImageClassifier(object):
    def __init__(self, arch, model_params, finetune = False, use_cuda = False, devs=[0]):
        self.arch = arch
        self.finetune = finetune
        self.use_cuda = use_cuda
        self.model_params = model_params
        self.devs = devs
        self.model_loaded = False
        rescale_size = 512
        crop_size = 448
        normalize = transforms.Normalize(mean=utils.IMAGENET_MEAN,
                                         std=utils.IMAGENET_STD)
        self.trans = transforms.Compose([
            transforms.Scale(rescale_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ])

    def load_model(self, arch, model_params, devs=[0], finetune = False, use_cuda = False):
        model = models.ResNet_FT(arch, downsample=False)
        if use_cuda:
            model = torch.nn.DataParallel(model, device_ids=devs).cuda()
            cudnn.benchmark = True
        checkpoint = torch.load(model_params)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def image_preprocessed(self, image):
        batch_imgs = torch.stack([self.trans(image)])
        return batch_imgs

    # import PIL image
    def classifyImage(self, image):
        if not self.model_loaded:
            self.model = self.load_model(self.arch, self.model_params, self.devs, self.finetune, self.use_cuda)
            self.model.eval()
            self.model_loaded = True
        input = self.image_preprocessed(image)

        if self.use_cuda:
            input = input.cuda()

        input_var = torch.autograd.Variable(input, volatile=True)

        output = self.model(input_var)

        pred = output.data.max(1)[1]
        m = torch.nn.Softmax()
        prop = m(output).data.max(1)[0]

        res = 0
        if self.use_cuda:
            res = pred.cpu().numpy()
            res_prop = prop.cpu().numpy()
        else:
            res = pred.numpy()
            res_prop = prop.numpy()

        return res[0][0], res_prop[0][0]

def import_test():
    print("welcome!")

def get_classifier():
    classifier = DrImageClassifier(args.arch, args.model, False, use_cuda, devs)
    return classifier

def main():

    classifier = DrImageClassifier(args.arch, args.model, False, use_cuda, devs)

    imgpath = 'sample/1/978_left.jpeg'
    img = Image.open(imgpath)

    idx,_ = classifier.classifyImage(img)

    print(idx)


if __name__ == '__main__':
    main()
