from torchvision.transforms.transforms import CenterCrop
from torchmetrics import JaccardIndex
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np                                                                                                   
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, models
import torch.cuda
import tensorflow as tf
import time



#data preparation

obj_classes = ['background', 'aeroplane', 'bicycle', 'bird',
                'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor', 'void/unlabelled']

#using Pascal VOC Dataset to get filenames and labels

sclass_path = '/content/drive/MyDrive/Python/PascalVOC2012/VOC2012/SegmentationClass/'
jpeg_path = '/content/drive/MyDrive/Python/PascalVOC2012/VOC2012/JPEGImages/'


test_set_labels = []
test_set_samples = []



#converting samples and labels to numpy array
def _remove_colormap(filename):
    return np.array(Image.open(filename))

#using text files to load filenames into lists
def loadDatasets(filename, imgarray):
    f = open(filename)
    for line in iter(f):
        line = line[0:15]
        imgarray.append(line)


loadDatasets('/content/drive/MyDrive/Python/labels/test_set.txt', test_set_labels)
loadDatasets('/content/drive/MyDrive/Python/samples/test_set.txt', test_set_samples)

batch_size = 10
image_size = 300


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
        sample = sample.type(torch.FloatTensor)
        
        sample = sample.permute(2, 0, 1)
        
        label = self.datalist_label[index]
        label = _remove_colormap(os.path.join(sclass_path, label))
        label = cv2.resize(label, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
        label = torch.tensor(label, dtype=torch.float32)
        label = label.type(torch.LongTensor)
        label[label > 20] = 21

        sample = sample, label
        

        return sample


#transforms for testing set
test_transform = transforms.Compose([
    transforms.CenterCrop(image_size),
    transforms.Resize(size=(image_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor()
])

def getDatasets():
    #dataset
    test_dataset = PascalVOCDataset(test_set_samples, test_set_labels, test_transform)
    #dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

test_loader = getDatasets()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE = "/content/drive/MyDrive/Python/models/resnet50_good_lr.pth"
resnet50fcn = models.segmentation.fcn_resnet50(num_classes=len(obj_classes)).to(device)
resnet50fcn.load_state_dict(torch.load(FILE))
resnet50fcn.eval().to(device)

#model testing using samples/test_set.txt and labels/test_set.txt
def test_accuracy(model):
    print("Testing accuracy of model...")
    with torch.no_grad():
        IoUs = []
        IoU_network_mean = 0
        num_correct = 0
        num_samples = 0
        n_class_correct = [0 for i in range(len(obj_classes))]
        n_class_samples = [0 for i in range(len(obj_classes))]

        for samples, labels in test_loader:
            samples = samples.to(device)
            labels = labels.to(device)
            samples = samples.type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.LongTensor)

            outputs = model(samples)['out']

            _, predictions = torch.max(outputs, 1)

            jaccard = JaccardIndex(num_classes=22, ignore_index=21, reduction='none').to(device)
            IoU = jaccard(predictions, labels).cpu()
            IoUs.append(np.array(IoU))

            #accuracy using number of correct pixels
            for i in range(5):
                for j in range(image_size):
                    for k in range(image_size):
                        label = labels[i][j][k].item()
                        pred = predictions[i][j][k].item()
                        if (label == pred):
                            n_class_correct[int(label)] += 1
                        n_class_samples[int(label)] += 1

        for i in range(21):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {obj_classes[i]}: {acc:.4f}%')
            num_correct += n_class_correct[i]
            num_samples += n_class_samples[i]

        acc = 100.0 * num_correct / num_samples
        print(f'accuracy of network = {acc:.4f}')


        #accuracy using IoU
        IoU_class_mean = [0 for i in range(len(obj_classes)) ]
        IoUs = np.array(IoUs)

        for i in range(len(IoUs)):
            for j in range(len(obj_classes)-1):
                IoU_class_mean[j] += IoUs[i][j]

        for i in range(len(obj_classes)-1):
            IoU_class_mean[i] /= len(IoUs)
            IoU_network_mean += IoU_class_mean[i]

        IoU_network_mean /= (len(obj_classes)-1)

        print("IoU of model by classes: ")
        for i in range(len(obj_classes)-1):
            print(f'Accuracy of {obj_classes[i]}: {IoU_class_mean[i]*100:.4f} %')
        
        print(f'IoU of Model: {IoU_network_mean*100:.4f} %')

test_accuracy(resnet50fcn)