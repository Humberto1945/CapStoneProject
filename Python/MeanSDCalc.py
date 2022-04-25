from PIL import Image
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
import math

mean0 = 0.
mean1 = 0.
mean2 = 0.
std0 = 0.
std1 = 0.
std2 = 0.

pixCount = 0

""" 
    Calculation of data normalization values based on the training set
"""

folder_path = '/content/drive/MyDrive/Python/PascalVOC2012/VOC2012/JPEGImages/'
train_set_list = '/content/drive/MyDrive/Python/samples/train_set.txt'
train_set = []

fi = open(train_set_list)
for line in iter(fi):
  line = line.strip()
  train_set.append(line)
fi.close()

for filename in train_set:
  # print(filename)
  im = np.array(Image.open(os.path.join(folder_path, filename)))

  mean0 += np.sum(im[:,:,0])
  mean1 += np.sum(im[:,:,1])
  mean2 += np.sum(im[:,:,2])
  # print(mean1)

  # print(im.shape[0])
  # print(im.shape[1])
  pixCount += (im.shape[0] * im.shape[1])

  # print("M0: " + str(mean0))
  # print("M1: " + str(mean1))
  # print("M2: " + str(mean2))

# print(pixCount)
# print(pixCount * 255)

mean0 /= (pixCount * 255)
mean1 /= (pixCount * 255)
mean2 /= (pixCount * 255)

print("Final M0: " + str(round(mean0, 3)))
print("Final M1: " + str(round(mean1, 3)))
print("Final M2: " + str(round(mean2, 3)))

print("\n")

for filename in train_set:
  im = np.array(Image.open(os.path.join(folder_path, filename)))

  for row in im:
    for pix in row:
      std0 += (((pix[0] / 255) - mean0) ** 2)
      std1 += (((pix[1] / 255) - mean1) ** 2)
      std2 += (((pix[2] / 255) - mean2) ** 2)

std0 = math.sqrt(std0 / pixCount)
std1 = math.sqrt(std1 / pixCount)
std2 = math.sqrt(std2 / pixCount)


print("Final SD0: " + str(round(std0, 3)))
print("Final SD1: " + str(round(std1, 3)))
print("Final SD2: " + str(round(std2, 3)))
