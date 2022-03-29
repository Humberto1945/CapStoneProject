import torch
import torch.nn as nn
from torchvision import datasets, models
import pascal_voc_dataloading as PV
import matplotlib.pyplot as plt


# #model training
#doesn't work right now because images need to be normalized
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet101 = models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=22).to(device)
#resnet50 = models.segmentation.deeplabv3_resnet50(pretrained=False)

learning_rate = 0.01
num_epochs = 5
image_size = 224
batch_size = 5

train_loader, test_loader = PV.getDatasets()

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)


def train_model(model, learning_rate, num_epochs):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(train_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            samples = samples.type(torch.FloatTensor)
            labels = labels.type(torch.LongTensor)

            #forward
            outputs = model(samples)['out']
            #labels = torch.empty(batch_size, dtype=torch.long).random_(5)
            loss = criterion(outputs, labels)

            #backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 500 == 0:
                print(f'epoch {epoch + 1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss:.4f}')

    print('Training Complete')

train_model(resnet101, learning_rate, num_epochs)
#train_model(resnet50, learning_rate, num_epochs)


