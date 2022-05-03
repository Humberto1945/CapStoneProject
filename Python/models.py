from torchvision.transforms.transforms import CenterCrop
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
from torchvision import datasets, models, utils
import torch.cuda
import tensorflow as tf
import torchvision.transforms.functional as TF
import time
from torch.utils.tensorboard import SummaryWriter
import sys
import copy
import random

"""

The models.py file contains the dataloading and model training sections of the project. 
The file uses the PascalVOC dataset images to train with. 
It is designed so that the files can be used inside Google Colab. 
The shared drive must be dragged and dropped into "My Drive" so that the files cn be accessed when mounting the drive onto Google Colab. 
Then, the cell where models.py is located can be edited to specify a name to save the model under. 
This model will be saved directly into the drive under the models/ subdirectory. 
This is the location where the other models which were trained during the project were saved. 
Before training, change the name of the directory where the loss and accuracy logs for tensorboard will be stored, 
so as to not save two logs under the same directory.

Args: none (datasets and model can be changed manually)

Returns: various log files and a trained model

"""

#writes to logs folder to plot graph in tensorboard
loss_writer = SummaryWriter("/content/drive/MyDrive/Python/logs_deeplab/ver_1_1_loss")
acc_writer = SummaryWriter("/content/drive/MyDrive/Python/logs_deeplab/ver_1_1_acc")

#using Pascal VOC Dataset to get filenames and labels

sclass_path = '/content/drive/MyDrive/Python/PascalVOC2012/VOC2012/SegmentationClass/'
jpeg_path = '/content/drive/MyDrive/Python/PascalVOC2012/VOC2012/JPEGImages/'

validation_set_labels = []
validation_set_samples = []
train_set_labels = []
train_set_samples = []


#using text files to load filenames into lists
def loadDatasets(filename, imgarray):
    f = open(filename)
    for line in iter(f):
        line = line[0:15]
        imgarray.append(line)

loadDatasets('/content/drive/MyDrive/Python/labels/train_set.txt', train_set_labels)
loadDatasets('/content/drive/MyDrive/Python/labels/validation_set.txt', validation_set_labels)
loadDatasets('/content/drive/MyDrive/Python/samples/train_set.txt', train_set_samples)
loadDatasets('/content/drive/MyDrive/Python/samples/validation_set.txt', validation_set_samples)

image_size = 512

class PascalVOCDataset(Dataset):
    def __init__(self, datalist_sample, datalist_label, transform, target_transform):
        self.datalist_sample = datalist_sample
        self.datalist_label = datalist_label
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.datalist_sample)

    def __getitem__(self, index):
        sample = self.datalist_sample[index]
        sample = np.array(Image.open(os.path.join(jpeg_path,sample)))
        sample = self.transform(sample)
        
        label = self.datalist_label[index]
        label = np.array(Image.open(os.path.join(sclass_path, label)))
        label[label > 20] = 21
        label = np.array(label)
        label = torch.tensor(label, dtype=torch.uint8)
        label = self.target_transform(label)

        label = torch.unsqueeze(label, 0)
        if random.randint(0, 1) > 0:
            sample = TF.hflip(sample)
            label = TF.hflip(label)
        if random.randint(0, 1) > 0:
            angle = random.randint(0, 4) * 90
            sample = TF.rotate(sample, angle)
            label = TF.rotate(label, angle)
        label = torch.squeeze(label, 0)
        
        return sample, label

        
#transforms for training and testing set
train_transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(image_size),
  transforms.ToTensor(),
  transforms.Normalize([0.457, 0.443, 0.407], [0.273, 0.269, 0.285])
])

target_transform = transforms.Compose([
  transforms.CenterCrop(image_size)                            
])


def getDatasets():
    #datasets
    train_dataset = PascalVOCDataset(train_set_samples, train_set_labels, train_transform, target_transform)
    val_dataset = PascalVOCDataset(validation_set_samples, validation_set_labels, train_transform, target_transform)
    #dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
    return train_loader, val_loader

train_loader, val_loader = getDatasets()


# #model training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


resnet50fcn = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=21).to(device)
resnet50deeplab = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=21).to(device)

#computed in class_weights.py
weights = torch.tensor([0.0230, 1.1018, 2.7316, 1.1537, 2.0097, 1.3983, 0.6186, 0.9153, 0.3955,
        2.2103, 1.0000, 1.3898, 0.4095, 0.7306, 0.9562, 0.2107, 2.4971, 1.1579,
        0.8685, 0.6411, 1.2548])
weights = weights.to(device)

def train_model(model, learning_rate, num_epochs):
    print("Training model. This will take a moment...")
    PATH = '/content/drive/MyDrive/Python/models/resnet50deeplab_ver_1_1.pth'
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=21)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    n_total_steps = len(train_loader)

    #training loop
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        train_losses = [0 for i in range(num_epochs)]
        epochs = []
        epochs.append(epoch)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            if phase == 'train':
                loader = train_loader
            else:
                loader = val_loader

            for i, (samples, labels) in enumerate(loader):
                samples = samples.to(device)
                labels = labels.to(device)
                samples = samples.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.LongTensor)
                
                
                #forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(samples)['out']
                    predictions = torch.argmax(outputs, 1)
                    loss = criterion(outputs, labels)

                    #backwards
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    #accumulates losses per epoch
                    train_losses[epoch] += loss.item()
                    running_loss += loss.item()
                    result = (predictions == labels)
                    running_correct += torch.sum(result)

                    
            if phase == 'train':
                epoch_loss = running_loss / len(train_loader)
                loss_writer.add_scalar("training loss", epoch_loss, epoch + 1)
                epoch_acc = running_correct.double() / (len(train_loader) * image_size * image_size * 10)
                acc_writer.add_scalar("training accuracy", epoch_acc, epoch + 1)
            else:
                epoch_loss = running_loss / len(val_loader)
                loss_writer.add_scalar("validation loss", epoch_loss, epoch + 1)
                epoch_acc = running_correct.double() / (len(val_loader) * image_size * image_size)
                acc_writer.add_scalar("validation accuracy", epoch_acc, epoch + 1)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            #deep copy the model
            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_weights)
                torch.save(model.state_dict(), PATH)

    
    print(f'Training with lr={learning_rate} and num_epochs={num_epochs} complete')



train_model(resnet50deeplab, learning_rate=0.001, num_epochs=200)
