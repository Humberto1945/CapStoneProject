from torchvision.transforms.transforms import CenterCrop
import torch
from torchvision import transforms
import numpy as np                                                                                                   
from PIL import Image
import os
import torch

"""

Computes the class weights for the provided training set by looping through the image labels and counting the pixels of each class occurance
then converts it to a tensor
Args: none (dataset and class list can be manually changed)
Returns: none (prints out tensor/array of class weights)

"""

#using Pascal VOC Dataset to get filenames and labels
obj_classes = ['background', 'aeroplane', 'bicycle', 'bird',
                'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor', 'void/unlabelled']
                
sclass_path = '/content/drive/MyDrive/Python/PascalVOC2012/VOC2012/SegmentationClass/'

image_size = 300
train_set_labels = []

#using text files to load filenames into lists
def loadDatasets(filename, imgarray):
    f = open(filename)
    for line in iter(f):
        line = line[0:15]
        imgarray.append(line)

loadDatasets('/content/drive/MyDrive/Python/labels/test_set.txt', train_set_labels)

for i in range(len(train_set_labels)):
    train_set_labels[i] = np.array(Image.open(os.path.join(sclass_path, train_set_labels[i])))
    train_set_labels[i] = torch.tensor(train_set_labels[i], dtype=torch.uint8)
    centercrop = transforms.CenterCrop(image_size)
    train_set_labels[i] = centercrop(train_set_labels[i])

class_frequency = [0 for i in range(len(obj_classes)-1)]
total_pixels = 0

for image in train_set_labels:
    for row in image:
        for pixel in row:
            if(pixel.item() == 255):
                continue
            class_frequency[pixel.item()] += 1
            total_pixels += 1

for i in range(len(class_frequency)):
    class_frequency[i] /= total_pixels

class_frequency = torch.tensor(class_frequency, dtype=torch.float32)
median = torch.median(class_frequency)

for i in range(len(class_frequency)):
    class_frequency[i] = median / class_frequency[i]

print(class_frequency)





