from math import ceil
from pickletools import uint8
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np                                                                                                   
from PIL import Image
from skimage.io import imshow
import os
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#data preparation

#using Pascal VOC Dataset to get filenames and labels
dic_path = 'PascalVOC2012/VOC2012/ImageSets/Main'

obj_classes = ['background', 'aeroplane', 'bicycle', 'bird',
                'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor', 'void/unlabelled']

segmentation_path = 'PascalVOC2012/VOC2012/ImageSets/Segmentation'
sclass_path = 'PascalVOC2012/VOC2012/SegmentationClass/'

validation_set = []
test_set = []
train_set = []



# def loadDatasets(filename, imgarray):
#     f = open(filename)
#     for line in iter(f):
#         line = line[0:15]
#         imgarray.append(line)

# loadDatasets('train_set.txt', train_set)
# loadDatasets('test_set.txt', test_set)
# loadDatasets('validation_set.txt', validation_set)


class PascalVOCDataset(Dataset):
    def __init__(self, datalist, transform):
        self.datalist = datalist
        self.transform = transform

    def _remove_colormap(filename):
        return np.array(Image.open(filename))


    
        

#converting segmentation labels
def _remove_colormap(filename):
    return np.array(Image.open(filename))

'''
for i in range(len(validation_set)):
    validation_set[i] = _remove_colormap(os.path.join(sclass_path, validation_set[i]))
#print(validation_set)

for i,image in enumerate(test_set):
    test_set[i] = _remove_colormap(os.path.join(sclass_path, image))
#print(test_set)
'''
for i,image in enumerate(train_set):
    train_set[i] = _remove_colormap(os.path.join(sclass_path, image))
    
    #print(train_set[i])
    #print()


#print(_remove_colormap(os.path.join(sclass_path, train_set[4])))

'''
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

'''

#model training

learning_rate = 0.01
num_epochs = 5
batch_size = 20
'''
for i in range(len(train_set)):
    train_set[i] = torch.tensor(train_set[i]).to(dtype=torch.uint8)
    print(train_set[i])'''

for i in range(len(train_set)):
    height, width = train_set[i].shape
    num_pixels = height * width
    #print(height, width)
    #print(num_pixels)

train_transform = transforms.Compose([
    
    transforms.Resize((300, 300), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()
])


#data = datasets.ImageFolder(root=sclass_path, transform=train_transform)
train_set = torch.tensor(train_set)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

#model = models.resnet50(pretrained=False)

