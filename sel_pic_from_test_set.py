# import shutil
# import os
#
#
#
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('data', metavar='./kaggle_dr/test', help='the kaggle dr test data')
# # parser.add_argument('label_csv', metavar='./trainLabels.csv', help='the labels for dr data')
# parser.add_argument('output_path', metavar='./kaggle_dr/data/test', help='the output res path')
#
# args = parser.parse_args()
#
#
# # 1. check directory exists or not, if not, create!
#
# import os
#
#
# root_path=args.output_path
# # label_file = args.label_csv
#
#
#
# for i in range(5):
#     dir = os.path.join(root_path,str(i))
#     if os.path.isdir(dir):
#         print("dir exist!")
#         pass
#     else:
#         print("dir not exists, create:")
#         os.makedirs(dir)
#
# # 2. read rsv file, move the photos to corresponding directories
#
# import pandas as pd
#
# labels = pd.read_csv(label_file, index_col=0)
#
# print(labels)
#
# from glob import glob
#
# # raw_pic_path = 'data/train_medium'
# raw_pic_path = args.data
#
# pics=glob('{}/*'.format(raw_pic_path))
# # pics=[os.path.basename(x).split('.')[0] for x in pics]
# print(pics)
#
# import shutil
#
# for pic in pics:
#     label = labels.loc[os.path.basename(pic).split('.')[0]].values[0]
#     shutil.copy(pic,os.path.join(root_path,str(label)))
#     print(os.path.join(root_path,str(label)))
#     print(label)
#
#
# # 3. Preprocess the data to medium data
#


import os
import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='./kaggle_dr/test', help='the kaggle dr test data')
parser.add_argument('output_path', metavar='./kaggle_dr/data/test', help='the output res path')

args = parser.parse_args()

root_path=args.output_path

raw_pic_path = args.data

for i in range(5):
    dir = os.path.join(root_path,str(i))
    if os.path.isdir(dir):
        print("dir exist!")
        pass
    else:
        print("dir not exists, create:")
        os.makedirs(dir)

images = [line.strip() for line in open('./val_images.txt', 'r')]
labels = [int(line.strip()) for line in open('./val_labels.txt', 'r')]

assert len(images) == len(labels)

print(len(labels))

images_index1 = []

for i, index in enumerate(labels):
    image = images[i] + '.jpeg'
    image = os.path.join(raw_pic_path,image)
    shutil.copy(image, os.path.join(root_path, str(index)))

