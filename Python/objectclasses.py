from math import ceil
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np                                                                                                   
from PIL import Image
from skimage.io import imshow
import os

obj_classes = ['background', 'aeroplane', 'bicycle', 'bird',
                'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor', 'void/unlabelled']

segmentation_path = 'PascalVOC2012/VOC2012/ImageSets/Segmentation'
sclass_path = 'PascalVOC2012/VOC2012/SegmentationClass/'
dic_path = 'PascalVOC2012/VOC2012/ImageSets/Main'

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

'''

#comparing palette with palette in a SegmentationClass image file
palette = np.reshape(palette, [-1,])
image_filepath = os.path.join(sclass_path, train_set[0])
impalette = Image.open(image_filepath).getpalette()
res = np.sum(impalette - palette)
print ("\nSum(impalette - palette) =  {}".format(res))
'''