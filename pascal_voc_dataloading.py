from ctypes import resize
from math import ceil
from pickletools import uint8
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


obj_classes = ['background', 'aeroplane', 'bicycle', 'bird',
                'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor', 'void/unlabelled']

segmentation_path = 'PascalVOC2012/VOC2012/ImageSets/Segmentation'

sclass_path = 'PascalVOC2012/VOC2012/SegmentationClass/'
jpeg_path = 'PascalVOC2012/VOC2012/JPEGImages/'

validation_set_labels = []
test_set_labels = []
train_set_labels = []
validation_set_samples = []
test_set_samples = []
train_set_samples = []

#converting segmentation labels
def _remove_colormap(filename):
    return np.array(Image.open(filename))

def loadDatasets(filename, imgarray):
    f = open(filename)
    for line in iter(f):
        line = line[0:15]
        imgarray.append(line)

loadDatasets('labels/train_set.txt', train_set_labels)
loadDatasets('labels/test_set.txt', test_set_labels)
loadDatasets('labels/validation_set.txt', validation_set_labels)
loadDatasets('samples/train_set.txt', train_set_samples)
loadDatasets('samples/test_set.txt', test_set_samples)
loadDatasets('samples/validation_set.txt', validation_set_samples)

print(train_set_labels[2])
print(train_set_samples[2])

# for i in range(len(train_set_samples)):
#     train_set_samples[i] = _remove_colormap(train_set_samples[i])
#     train_set_labels[i] = _remove_colormap(train_set_)

learning_rate = 0.01
num_epochs = 5
batch_size = 20

class PascalVOCDataset(Dataset):
    def __init__(self, datalist_sample, datalist_label, transform):
        self.datalist_sample = datalist_sample
        self.datalist_label = datalist_label
        self.transform = transform

    def __len__(self):
        return len(self.datalist_sample)

    def __getitem__(self, index):
        sample = self.datalist_sample[index]
        sample = _remove_colormap(os.path.join(jpeg_path,sample))
        #sample = Image.fromarray(np.uint8(sample))
        #sample = resize(224, 224)
        # sample = cv2.imread(os.path.join(jpeg_path, sample))
        # sample = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        label = self.datalist_label[index]
        label = _remove_colormap(os.path.join(sclass_path, label))
        #label = Image.fromarray(np.uint8(label))
        #label = resize(224, 224)
        # label = cv2.imread(os.path.join(sclass_path, label))
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        
        return sample, label
    
train_transform = transforms.Compose([
    #transforms.Resize(300, interpolation=2)
    transforms.ToPILImage(),
    transforms.Resize((300, 300), transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((300, 300), transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor()
])

train_dataset = PascalVOCDataset(train_set_samples, train_set_labels, transform=train_transform)
print(train_dataset[5])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

# for sample, label in train_loader:
#     print(sample.shape)

# val_dataset = PascalVOCDataset(validation_set_samples, validation_set_labels, test_transform)
# test_dataset = PascalVOCDataset(test_set_samples, test_set_labels, test_transform)
       


# #model training





# for i in range(len(train_set_samples)):
#     height, width = train_set_samples[i].shape
#     num_pixels = height * width
    #print(height, width)
    #print(num_pixels)



#model = models.resnet50(pretrained=False)

