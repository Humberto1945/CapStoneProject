from math import ceil
import random
import numpy as np                                                                                                   
import os

"""

*INSERT WHAT IT DOES HERE*
Args:
Returns:

"""

dic_path = 'PascalVOC2012/VOC2012/ImageSets/Main'

obj_classes = ['background', 'aeroplane', 'bicycle', 'bird',
                'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor', 'void/unlabelled']

train_set = []
validation_set = []

segmentation_path = 'PascalVOC2012/VOC2012/ImageSets/Segmentation'
sclass_path = 'PascalVOC2012/VOC2012/SegmentationClass'

def make_datasets():
    files = os.listdir(segmentation_path)
    for file in files:
        if ('trainval' not in file):
            if ('train' in file):
                f = open(os.path.join(segmentation_path, file))
                for line in iter(f):
                    line = line[0:11]
                    train_set.append(line + ".PNG")
            else:
                f = open(os.path.join(segmentation_path, file))
                for line in iter(f):
                    line = line[0:11]
                    validation_set.append(line + ".PNG")

make_datasets()

test_set = random.sample(validation_set, ceil(len(validation_set)*.5))

val_set = []
for i in range(len(validation_set)):
    if (validation_set[i] in test_set):
        continue
    val_set.append(validation_set[i])

print(len(validation_set))
print(len(val_set))
val_set = random.sample(val_set, len(val_set))


val_set = np.array(val_set)
test_set = np.array(test_set)
train_set = np.array(train_set)

np.savetxt("validation_set.txt", val_set, delimiter=",", fmt='%s')
np.savetxt("test_set.txt", test_set, delimiter=",", fmt='%s')
np.savetxt("train_set.txt", train_set, delimiter=",", fmt='%s')
