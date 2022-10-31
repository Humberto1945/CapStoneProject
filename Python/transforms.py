from torchvision import transforms

image_size = 512

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

test_transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(image_size),
  transforms.ToTensor()
])
PIL_transform = transforms.Compose([
  transforms.ToPILImage()
])
tensor_transform = transforms.Compose([
  transforms.ToTensor()
])