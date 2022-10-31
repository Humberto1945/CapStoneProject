import torch
import PIL
from PIL import Image
import transforms
import numpy as np
import os
from torchvision import models
import pixels



def load_image(image, mask):
    imagePath = "/content/drive/MyDrive/Python/PascalVOC2012/VOC2012/JPEGImages/"
    maskPath = "/content/drive/MyDrive/Python/PascalVOC2012/VOC2012/SegmentationClass/"
    image = Image.open(os.path.join(imagePath, image))
    mask = Image.open(os.path.join(maskPath, mask))
    image = np.array(image)
    mask = np.array(mask)
    image = transforms.train_transform(image)
    mask = transforms.test_transform(mask)
    image = torch.unsqueeze(image, 0)
    mask = torch.unsqueeze(mask, 0)
    return image, mask


def create_mask(model, LOCAL_FILE, image_file, mask_file, view):
    model.load_state_dict(torch.load(LOCAL_FILE, map_location=device))
    model.eval().to(device)
    PATH = "C:/Users/aless/Documents/VSU_Classes/Spring 2022/Senior Seminar/PascalVOC2012/VOC2012/JPEGImages/"
    PATH2 = "C:/Users/aless/Documents/VSU_Classes/Spring 2022/Senior Seminar/PascalVOC2012/VOC2012/SegmentationClass/"
    
    image = Image.open(os.path.join(PATH, image_file))
    label = Image.open(os.path.join(PATH2, mask_file))
    if view == True:
        image.show()
        label.show()
    image = np.array(image)
    label = np.array(label)
    image = transforms.train_transform(image)
    label = transforms.test_transform(label)
    image = torch.unsqueeze(image, 0)
    label = torch.unsqueeze(label, 0)
    output = model(image)['out']
    prediction = torch.argmax(output, 1)
    result = (prediction == label)
    print(torch.sum(result))
    return prediction, image, label

    
def show_mask(prediction, image, mask, rotate):
    image = pixels.change_pixels(image, image_size, prediction)
    image = torch.squeeze(image)
    image = transforms.PIL_transform(image)
    if rotate == True:
        image = image.rotate(270)
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    result = (prediction == mask)
    print(torch.sum(result))
    image.show()
    
def majority(prediction1, prediction2, prediction3):
    pred1 = np.array(prediction1)
    pred2 = np.array(prediction2)
    pred3 = np.array(prediction3)

    pred1 = pred1.flatten()
    pred2 = pred2.flatten()
    pred3 = pred3.flatten()

    changes = np.where(pred2 == pred3)
    for i in range(0, changes[0].size):
        pred1[changes[0][i]] = pred2[changes[0][i]]

    pred1 = np.reshape(pred1, (1,512,512))
    pred1 = transforms.tensor_transform(pred1)
    return pred1

def acc(pred, mask):
    result = (pred == mask)
    return result

device = torch.device('cpu')
image_size = 512

image_file = "2007_006900.jpg"
mask_file = "2007_006900.png"

resnet50fcn = models.segmentation.fcn_resnet50(weights=None, progress=True, num_classes=21).to(device)
resnet50deeplab =  models.segmentation.deeplabv3_resnet50(weights=None, progress=True, num_classes=21).to(device)
resnet101deeplab = models.segmentation.deeplabv3_resnet101(weights=None, progress=True, num_classes=21).to(device)

resnet50fcn_path = "resnet50_ver_1_3.pth"   
resnet101deeplab_path = "resnet101deeplab_ver_3.pth"
resnet50deeplab_path = "resnet50deeplab_ver_1_1.pth"

prediction1, image, mask = create_mask(resnet101deeplab, resnet101deeplab_path, image_file, mask_file, True)
prediction2, _, _ = create_mask(resnet50fcn, resnet50fcn_path, image_file, mask_file, False)
prediction3, _, _ = create_mask(resnet50deeplab, resnet50deeplab_path, image_file, mask_file, False)

pred_m = majority(prediction1, prediction2, prediction3)
pred_m = pred_m.view(1,512,512)

show_mask(pred_m, image, mask, True)
show_mask(prediction1, image, mask, False)
show_mask(prediction2, image, mask, False)
show_mask(prediction3, image, mask, False)