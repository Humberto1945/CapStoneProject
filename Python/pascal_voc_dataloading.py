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

print("Samples",train_set_labels[2])
print("Labels", train_set_samples[2])

learning_rate = 0.01
num_epochs = 5
batch_size = 5
image_size = 300

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(image_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToPILImage(),
    
])

test_transform = transforms.Compose([
    transforms.Resize(size=(image_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor()
])

class PascalVOCDataset(Dataset):
    def __init__(self, datalist_sample, datalist_label):
        self.datalist_sample = datalist_sample
        self.datalist_label = datalist_label
        
    def __len__(self):
        return len(self.datalist_sample)

    def __getitem__(self, index):
        sample = self.datalist_sample[index]
        sample = _remove_colormap(os.path.join(jpeg_path,sample))
        sample = cv2.resize(sample, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
        train_transform(sample)
        sample = torch.tensor(sample, dtype=torch.uint8)
        sample = sample[:3]
        
        label = self.datalist_label[index]
        label = _remove_colormap(os.path.join(sclass_path, label))
        label = cv2.resize(label, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
        train_transform(label)
        label = torch.tensor(label, dtype=torch.uint8)
        
        return sample, label
    

train_dataset = PascalVOCDataset(train_set_samples, train_set_labels)

print(train_dataset[5])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)
print(type(samples))
print(type(labels))

# image = Image.open(os.path.join(jpeg_path, train_set_samples[0]))
# plt.subplot(311)
# plt.imshow(image)
# plt.subplot(312)
# plt.imshow(train_transform(image))
# plt.show()
    
# #model training
#doesn't work right now because images need to be normalized
model = models.resnet50(pretrained=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (samples, labels) in enumerate(train_loader):
        samples = samples.to(device)
        labels = labels.to(device)

        #forward
        outputs = model(samples)
        loss = criterion(outputs, labels)

        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 500 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss:.4f}')

print('Training Complete')