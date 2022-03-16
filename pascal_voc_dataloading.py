<<<<<<< HEAD
from math import ceil
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
import glob
from skimage.io import imshow
import tensorflow as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#data preparation
#using Pascal VOC Dataset to get fimenames and labels
dic_path = 'PascalVOC2012/VOC2012/ImageSets/Main'

obj_classes = ['background', 'aeroplane', 'bicycle', 'bird',
                'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor', 'void/unlabelled']

train_set = []
validation_set = []

segmentation_path = 'PascalVOC2012/VOC2012/ImageSets/Segmentation'
sclass_path = 'PascalVOC2012/VOC2012/SegmentationClass'

def make_datasets():
    files = os.listdir(segmentation_path)
    for file in files:
        if ('trainval' not in file):
            if ('train' in file):
                f = open(os.path.join(segmentation_path, file))
                for line in iter(f):
                    line = line[0:11]
                    train_set.append(line + ".PNG")
            else:
                f = open(os.path.join(segmentation_path, file))
                for line in iter(f):
                    line = line[0:11]
                    validation_set.append(line + ".PNG")

make_datasets()
#print(train_set)
#print(validation_set)
#print(len(validation_set)/2)
test_set = random.sample(validation_set, ceil(len(validation_set)*.5))
#print(len(test_set))
#print(test_set[0])
val_set = []
for i in range(len(validation_set)):
    if (validation_set[i] in test_set):
        continue
    val_set.append(validation_set[i])

#The sets are:
#train_set
#test_set
#val_set
count = 0


#print(val_set)

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    rgbmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        rgbmap[i] = np.array([r, g, b])

    rgbmap = rgbmap/255 if normalized else rgbmap
    return rgbmap


palette = color_map(N=256)
ignore = palette[255]

#prints object classes and associated rgb colors
colors = np.array(palette)
colors = colors[0:21, :]
colors = np.append(colors, [ignore], 0)

def class_colors(colors, obj_classes):
    for i in range(len(obj_classes)-1):
        print('{:>3d}: {:<20} rgb: {}'.format(i, obj_classes[i], colors[i]))
    print('{:>3d}: {:<20} rgb: {}'.format(255, obj_classes[len(obj_classes)-1], colors[len(colors)-1]))

class_colors(colors, obj_classes)



#comparing palette with palette in a SegmentationClass image file
palette = np.reshape(palette, [-1,])
image_filepath = os.path.join(sclass_path, train_set[0])
impalette = Image.open(image_filepath).getpalette()
res = np.sum(impalette - palette)
print ("\nSum(impalette - palette) =  {}".format(res))


#converting segmentation labels
def _remove_colormap(filename):
    return np.array(Image.open(filename))


for i in range(len(train_set)):
    #print(train_set[i])
    train_set[i] = _remove_colormap(os.path.join(sclass_path, train_set[i]))
    #print(train_set[i])
    #print("\n")
for i in range(len(val_set)):
    validation_set[i] = _remove_colormap(os.path.join(sclass_path, val_set[i]))

for i in range(len(test_set)):
    test_set[i] = _remove_colormap(os.path.join(sclass_path, test_set[i]))



mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'SegmentationClass' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
}


#model training
model = models.resnet50(pretrained=False)

=======
from math import ceil
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
import glob
from skimage.io import imshow
import tensorflow as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#data preparation
#using Pascal VOC Dataset to get fimenames and labels
dic_path = 'PascalVOC2012/VOC2012/ImageSets/Main'

obj_classes = ['background', 'aeroplane', 'bicycle', 'bird',
                'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor', 'void/unlabelled']

train_set = []
validation_set = []

segmentation_path = 'PascalVOC2012/VOC2012/ImageSets/Segmentation'
sclass_path = 'PascalVOC2012/VOC2012/SegmentationClass'

def make_datasets():
    files = os.listdir(segmentation_path)
    for file in files:
        if ('trainval' not in file):
            if ('train' in file):
                f = open(segmentation_path + "/" + file)
                for line in iter(f):
                    line = line[0:11]
                    train_set.append(line + ".PNG")
            else:
                f = open(segmentation_path + "/" + file)
                for line in iter(f):
                    line = line[0:11]
                    validation_set.append(line + ".PNG")

make_datasets()
#print(train_set)
#print(validation_set)
#print(len(validation_set)/2)
test_set = random.sample(validation_set, ceil(len(validation_set)*.5))
#print(len(test_set))
#print(test_set[0])
val_set = []
for i in range(len(validation_set)):
    if (validation_set[i] in test_set):
        continue
    val_set.append(validation_set[i])

#print(val_set)

def _remove_colormap(filename):
    return np.array(Image.open(filename))

'''
for i in range(10):
    print(train_set[i])
    print(_remove_colormap(os.path.join(sclass_path, train_set[i])))
    print("\n")
'''


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

palette = color_map(N=256)

def color_map_info(palette, obj_classes):
    print("Class color map and palette = {r, g, b}")
    for i in range (0, 21*3, 3):
        print ('# {:>3d}: {:<20} (R,G,B) = {},{},{}'.format(int(i/3), obj_classes[int(i/3)], palette[i], palette[i+1],palette[i+2]))
    i = 255 * 3
    print ('# {:>3d}: {:<20} (R,G,B) = {},{},{}'.format(int(i/3), obj_classes[21], palette[i], palette[i+1],palette[i+2]))

palette = np.reshape(palette, [-1,])
color_map_info(palette, obj_classes)

annotation = sclass_path + "/" + train_set[0]
impalette = Image.open(annotation).getpalette()

res = np.sum(impalette - palette)
print ("\nSum(impalette - palette) =  {}".format(res))


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'SegmentationClass' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
}


#model training
model = models.resnet50(pretrained=False)

>>>>>>> e1a149f4358cf4d4320f29155129c28082c9a78c
