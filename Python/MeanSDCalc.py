from PIL import Image
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms

mean1 = 0.
mean2 = 0.
mean3 = 0.
std1 = 0.
std2 = 0.
std3 = 0.

files = 0
# pixCount = 0

""" 
    The JPEGImages folder contains 17125 images, but we do not use that many.
    For the time being the mean is calculated based on all of them
"""

folder_path = '/content/drive/MyDrive/Python/PascalVOC2012/VOC2012/JPEGImages'

# file_count = sum(len(files) for _, _, files in os.walk(folder_path))
# file_count = len(os.listdir(folder_path))
# print(file_count)

for filename in os.listdir(folder_path):
  # print(filename)
  im = Image.open(os.path.join(folder_path, filename))

  convertTensor = transforms.ToTensor()
  # convertSize = transforms.CenterCrop(512)

  # adding the size conversion greatly decreases the mean due to added empty pixels

  # im = convertSize(im)
  imTensor = convertTensor(im)

  files += 1

  # print(mean)

  # print(imTensor)

  # print(imTensor[:,:,0])
  # print(imTensor[:,:,1])
  # print(imTensor[:,:,2])
  
  # print(imTensor.shape[1])
  # print(imTensor.shape[2])
  
  mean1 += torch.mean(imTensor[:,:,0])
  mean2 += torch.mean(imTensor[:,:,1])
  mean3 += torch.mean(imTensor[:,:,2])

  std1 += torch.std(imTensor[:,:,0])
  std2 += torch.std(imTensor[:,:,1])
  std3 += torch.std(imTensor[:,:,2])

  # pixCount += imTensor.shape[1] * imTensor.shape[2]
  # print(pixCount)

  # print("M1: " + str(mean1.item()))
  # print("M2: " + str(mean2.item()))
  # print("M3: " + str(mean3.item()))

  # print("SD1: " + str(std1.item()))
  # print("SD2: " + str(std2.item()))
  # print("SD3: " + str(std3.item()))

mean1 /= files
mean2 /= files
mean3 /= files

std1 /= files
std2 /= files
std3 /= files

print("Final M1: " + str(round(mean1.item(), 3)))
print("Final M2: " + str(round(mean2.item(), 3)))
print("Final M3: " + str(round(mean3.item(), 3)))

print("\n")

print("Final SD1: " + str(round(std1.item(), 3)))
print("Final SD2: " + str(round(std2.item(), 3)))
print("Final SD3: " + str(round(std3.item(), 3)))