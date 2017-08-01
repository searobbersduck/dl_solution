import argparse
import os
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='dir', help='path to dataset')
parser.add_argument('-save', metavar='dir', default='./', help='path to save processed files')
parser.add_argument('-labels', metavar='labels', default='./class_1.txt', help='the txt files for images')

args = parser.parse_args()

fs = glob('{}/*'.format(args.data))
fs = [os.path.basename(f) for f in fs]


imgs = []
with open(args.labels) as f:
    for line in f:
        imgs.append(line.strip('\n'))


import scipy.misc

import numpy as np

import preprocessing

from PIL import Image

import skimage
from skimage.filters import threshold_otsu

def image_processing_and_save(imgpath, savepath, imgname):
    img = scipy.misc.imread(imgpath)
    img = img.astype(np.float32)
    img /= 255
    img_ahe = preprocessing.channelwise_ahe(img)
    pilImage = Image.fromarray(skimage.util.img_as_ubyte(img_ahe))
    pilImage.save(os.path.join(savepath, imgname))

def image_processing_and_save_segmentation(imgpath):
    img = scipy.misc.imread(imgpath)
    img = img.astype(np.float32)
    img /= 255
    img_ahe = preprocessing.channelwise_ahe(img)
    pilImage = Image.fromarray(skimage.util.img_as_ubyte(img_ahe))
    pilImage.save('segmentation/{}'.format(os.path.basename(imgpath)))

# for i in imgs:
#     if i in fs:
#         path = os.path.join(args.data, i)
#         print(i)
#         image_processing_and_save(path, args.save, i)


imgpath = './368_left_1024.png'
img = scipy.misc.imread(imgpath)
img = img.astype(np.float32)
img /= 255
img_ahe = preprocessing.channelwise_ahe(img)
pilImage = Image.fromarray(skimage.util.img_as_ubyte(img_ahe))
pilImage.save('368_left_1024_ahe_0.1.png')

# jpgroot = '/Users/zhangweidong03/Code/dl/pytorch/github/pi/piwise/MAdata_patches/images'
# files = os.listdir('/Users/zhangweidong03/Code/dl/pytorch/github/pi/piwise/MAdata_patches/images')
# files.sort()
# for f in files:
#     ff = os.path.join(root, f)
#     image_processing_and_save_segmentation(ff)