import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np                                                                                                   
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt


#data preparation

obj_classes = ['background', 'aeroplane', 'bicycle', 'bird',
                'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor', 'void/unlabelled']
#using Pascal VOC Dataset to get filenames and labels

sclass_path = 'PascalVOC2012/VOC2012/SegmentationClass/'
jpeg_path = 'PascalVOC2012/VOC2012/JPEGImages/'

validation_set_labels = []
validation_set_samples = []
test_set_labels = []
train_set_labels = []
test_set_samples = []
train_set_samples = []

#converting samples and labels to numpy array
def _remove_colormap(filename):
    return np.array(Image.open(filename))

#using text files to load filenames into lists
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

batch_size = 5
image_size = 224


class PascalVOCDataset(Dataset):
    def __init__(self, datalist_sample, datalist_label, transform):
        self.datalist_sample = datalist_sample
        self.datalist_label = datalist_label
        self.transform = transform
        
    def __len__(self):
        return len(self.datalist_sample)

    def __getitem__(self, index):
        sample = self.datalist_sample[index]
        sample = _remove_colormap(os.path.join(jpeg_path,sample))/255.0
        sample = cv2.resize(sample, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
        sample = torch.tensor(sample, dtype=torch.float32)
        mean = torch.mean(sample)
        std = torch.std(sample)
        sample = (sample-mean)/std
        # norm = transforms.Normalize(mean, std)
        # norm(sample)
        
        #sample = sample.permute(2, 0, 1)
        
        label = self.datalist_label[index]
        label = _remove_colormap(os.path.join(sclass_path, label))
        label = cv2.resize(label, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
        label = torch.tensor(label, dtype=torch.float32)
        # mean = torch.mean(label)
        # std = torch.std(label)
        # label = (label-mean)/std
        label[label==255] = len(obj_classes)

        sample = sample, label

        return sample



#transforms for training and testing set
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(image_size), interpolation=transforms.InterpolationMode.BILINEAR),
    #transforms.Normalize()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(image_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor()
])

def getDatasets():
    #datasets
    train_dataset = PascalVOCDataset(train_set_samples, train_set_labels, train_transform)
    test_dataset = PascalVOCDataset(test_set_samples, test_set_labels, test_transform)
    sample, label = train_dataset[5]
    print(label)
    #sample.transpose_(0, 2)
    fig = plt.figure()  
    a = fig.add_subplot(1,2,1)
    plt.imshow(sample)

    a = fig.add_subplot(1,2,2)
    plt.imshow(label)

    plt.show()
    #dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

train_loader, test_loader = getDatasets()

# examples = iter(train_loader)
# samples, labels = examples.next()
# print(samples.shape, labels.shape)



