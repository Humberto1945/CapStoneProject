import data_loading
import torch
import numpy as np
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from matplotlib.pyplot import imshow



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
        sample = Image.open(os.path.join(data_loading.jpeg_path,sample))
        if index == 1 or index == 3:
            imshow(np.asarray(sample))
        sample = np.array(sample)
        sample = self.transform(sample)
        
        label = self.datalist_label[index]
        label = Image.open(os.path.join(data_loading.sclass_path, label))
        if index == 1 or index == 3:
            imshow(np.asarray(label))
        label = np.array(label)
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
        if index == 1 or index == 3:
            transform = T.ToPILImage()
            s = transform(sample)
            l = transform(label)
            imshow(np.asarray(s))
            imshow(np.asarray(l))
        
        return sample, label

