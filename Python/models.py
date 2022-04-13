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
import time
from torch.utils.tensorboard import SummaryWriter
import sys

#writes to logs folder to plot graph in tensorboard
writer = SummaryWriter("/content/drive/MyDrive/Python/logs/PascalVOC")

#data preparation

obj_classes = ['background', 'aeroplane', 'bicycle', 'bird',
                'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor', 'void/unlabelled']

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

batch_size = 10
image_size = 300

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
        #sample = cv2.resize(sample, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
        # sample = torch.tensor(sample, dtype=torch.float32)
        # sample = sample.type(torch.FloatTensor)
        # mean = torch.mean(sample)
        # std = torch.std(sample)
        # sample = (sample-mean)/std
        # norm = transforms.Normalize(mean, std)
        # norm(sample)
        sample = self.transform(sample)
        
        label = self.datalist_label[index]
        label = np.array(Image.open(os.path.join(sclass_path, label)))
        #label = cv2.resize(label, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
        label[label > 20] = 21
        label = self.target_transform(label)
        label = torch.squeeze(label)
        

        return sample, label

        

#transforms for training and testing set
train_transform = transforms.Compose([
                                      transforms.ToPILImage(),
                                      transforms.CenterCrop(image_size),
                                      transforms.ToTensor(),

                                      #these are just placeholder values for testing
                                      transforms.Normalize([0.411, 0.412, 0.412], [0.187, 0.189, 0.190])
])

target_transform = transforms.Compose([
                                       transforms.ToPILImage(),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor()                              
])


def getDatasets():
    #datasets
    train_dataset = PascalVOCDataset(train_set_samples, train_set_labels, train_transform, target_transform)
    
    #dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

train_loader = getDatasets()

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

# #model training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#resnet101 = models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=22).to(device)
resnet50fcn = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=22).to(device)
#resnet50deeplab = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=22).to(device)

learning_rate = 0.001
num_epochs = 200

def train_model(model, learning_rate, num_epochs):
    print("Training model. This will take a moment...")
    criterion = nn.CrossEntropyLoss(ignore_index=21)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #training loop
    n_total_steps = len(train_loader)
    train_losses = [0 for i in range(num_epochs)]
    epochs = []

    since = time.time()
    running_loss = 0.0
    running_correct = 0
    
    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(train_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            samples = samples.type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.LongTensor)
            
            
            #forward
            outputs = model(samples)['out']
            loss = criterion(outputs, labels)

            #backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #accumulates losses per epoch
            train_losses[epoch] += loss.item()
            running_loss += loss.item()

            #keeps track of progress in training by printing epoch and associated loss
            if (i+1) % n_total_steps == 0:
                print(f'epoch {epoch + 1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')
                writer.add_scalar("training loss", running_loss / n_total_steps, epoch * n_total_steps + i)
                train_losses[epoch] /= n_total_steps
                epochs.append(epoch)
                running_loss = 0.0
                
    #keeps track of time elapsed in training
    #prints when training is complete and saves the model to specified path
    time_elapsed = time.time() - since 
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    PATH = './resnet50_good_lr.pth'
    torch.save(resnet50fcn.state_dict(), PATH)
    
    #plots the graph using average loss per epoch using matplotlib
    plt.plot(epochs, train_losses, "g", label="Training Loss")
    plt.title(f'Loss VS Epoch Large Learning Rate = {learning_rate}')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


print(f'Training with lr={learning_rate}, batch_size={batch_size} and num_epochs={num_epochs}')
train_model(resnet50fcn, learning_rate, num_epochs)