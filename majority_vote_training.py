import torch
from pixels import change_pixels
import transforms
import numpy as np                                                                                                   
import torch
from torchvision import models
import torch.cuda
from PIL import Image
import os
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def majority():
    #install untrained models
    resnet50fcn = models.segmentation.fcn_resnet50(weights=None, progress=True, num_classes=21).to(device)
    resnet50deeplab = models.segmentation.deeplabv3_resnet50(weights=None, progress=True, num_classes=21).to(device)
    resnet101deeplab = models.segmentation.deeplabv3_resnet101(weights=None, progress=True, num_classes=21).to(device)

    #load trained model state dicts into untrained models
    resnet50deeplab.load_state_dict(torch.load("/content/drive/MyDrive/Python/models/resnet50deeplab_ver_1_2.pth"))
    resnet50fcn.load_state_dict(torch.load("/content/drive/MyDrive/Python/models/resnet50_ver_1_3.pth"))
    resnet101deeplab.load_state_dict(torch.load("/content/drive/MyDrive/Python/models/resnet101deeplab_ver_3.pth"))
    resnet50deeplab.eval()
    resnet50fcn.eval()
    resnet101deeplab.eval()

    #run image through model and get predictions
    image, mask = load_image("2007_001825.jpg", "2007_001825.png")
    image = image.to(device)
    mask = mask.to(device)
    resnet50deeplab_outputs = resnet50deeplab(image)['out']
    resnet50fcn_outputs = resnet50fcn(image)['out']
    resnet101deeplab_outputs = resnet101deeplab(image)['out']

    resnet50deeplab_predictions = torch.argmax(resnet50deeplab_outputs, 1)
    resnet50fcn_predictions = torch.argmax(resnet50fcn_outputs, 1)
    resnet101deeplab_predictions = torch.argmax(resnet101deeplab_outputs, 1)

    result_101deeplab = (resnet101deeplab_predictions == mask)
    result_50deeplab = (resnet50deeplab_predictions == mask)
    result_50fcn = (resnet50fcn_predictions == mask)

    print("101 Deeplab: ", torch.sum(result_101deeplab))
    print("50 Deeplab: ", torch.sum(result_50deeplab))
    print("50 FCN: ", torch.sum(result_50fcn))


    #majority vote section
    resnet50deeplab_predictions = resnet50deeplab_predictions.cpu()
    resnet50fcn_predictions = resnet50fcn_predictions.cpu()
    resnet101deeplab_predictions = resnet101deeplab_predictions.cpu()

    resnet50deeplab_predictions = np.array(resnet50deeplab_predictions)
    resnet50fcn_predictions = np.array(resnet50fcn_predictions)
    resnet101deeplab_predictions = np.array(resnet101deeplab_predictions)

    resnet50deeplab_predictions.flatten()
    resnet50fcn_predictions.flatten()
    resnet101deeplab_predictions.flatten()


    changes = np.where(resnet50deeplab_predictions == resnet50fcn_predictions)

    for i in range(0, changes[0].size):
        resnet101deeplab_predictions[changes[0][i]] = resnet50deeplab_predictions[changes[0][i]]

    # resnet101d = np.reshape(resnet101deeplab_predictions, (1,512,512))
    resnet101d = transforms.tensor_transform(resnet101deeplab_predictions)
    # resnet50d = transforms.tensor_transform(resnet50deeplab_predictions)
    # resnet50f = transforms.tensor_transform(resnet50fcn_predictions)
    resnet101d = resnet101d.permute(1,0,2)
    image_m = change_pixels(image=image, image_size=512, prediction=resnet101d)
    image_m = torch.squeeze(image_m)
    image_m = transforms.PIL_transform(image_m)

    # image_50f = change_pixels(image=image, image_size=512, prediction=resnet50f)
    # image_50f = torch.squeeze(image_50f)
    # image_50f = transforms.PIL_transform(image_50f)

    # resnet101d = resnet101d.to(device)
    # result_resnet101d = (resnet101d == mask)
    # print("101 Deeplab: ", torch.sum(result_resnet101d))
    return image_m




