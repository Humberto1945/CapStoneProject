from torchvision import models
import torch
import model_training
import SegNet

# choose model
# resnet50fcn = models.segmentation.fcn_resnet50(weights=None, progress=True, num_classes=21).to(model_training.device)
# resnet50deeplab = models.segmentation.deeplabv3_resnet50(weights=None, progress=True, num_classes=21).to(model_training.device)
resnet101deeplab = models.segmentation.deeplabv3_resnet101(weights=None, progress=True, num_classes=21).to(model_training.device)
# segnet = SegNet.SegNet(3, 21).to(model_training.device);

#computed in class_weights.py
weights = torch.tensor([0.0230, 1.1018, 2.7316, 1.1537, 2.0097, 1.3983, 0.6186, 0.9153, 0.3955,
        2.2103, 1.0000, 1.3898, 0.4095, 0.7306, 0.9562, 0.2107, 2.4971, 1.1579,
        0.8685, 0.6411, 1.2548])
weights = weights.to(model_training.device)


model_training.train_model(resnet101deeplab, learning_rate=0.001, num_epochs=200, weights=weights)
