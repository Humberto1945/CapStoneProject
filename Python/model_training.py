import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import PascalVOC
import SegNet
import data_loading
import transforms
from torch.utils.data import Dataset, DataLoader
import copy
import matplotlib.pyplot as plt

'''
Every time this program is run, do the following to prevent overwriting:
Change path of log_writer and acc_writer
Change path of PATH where the model is being saved
'''

#writes to logs folder to plot graph in tensorboard
loss_writer = SummaryWriter("/content/drive/MyDrive/Python/logs_deeplab_101/ver_3_loss")
acc_writer = SummaryWriter("/content/drive/MyDrive/Python/logs_deeplab_101/ver_3_acc")
PATH = '/content/drive/MyDrive/Python/models/resnet101deeplab_ver_3.pth'

#using Pascal VOC Dataset to get filenames and labels
image_size = 512

# #model training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model_with_validation(model, learning_rate, num_epochs, weights):
    train_set_labels = []
    train_set_samples = []

    data_loading.loadDatasets('/content/drive/MyDrive/Python/labels/train_set.txt', train_set_labels)
    data_loading.loadDatasets('/content/drive/MyDrive/Python/samples/train_set.txt', train_set_samples)
    validation_set_labels = []
    validation_set_samples = []
    data_loading.loadDatasets('/content/drive/MyDrive/Python/labels/validation_set.txt', validation_set_labels)
    data_loading.loadDatasets('/content/drive/MyDrive/Python/samples/validation_set.txt', validation_set_samples)
    val_dataset = PascalVOC.PascalVOCDataset(validation_set_samples, validation_set_labels, transforms.train_transform, transforms.target_transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
    train_dataset = PascalVOC.PascalVOCDataset(train_set_samples, train_set_labels, transforms.train_transform, transforms.target_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
    print("Training model. This will take a moment...")
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=21)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    

    #training loop
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        train_losses = [0 for i in range(num_epochs)]
        epochs = []
        epochs.append(epoch)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            if phase == 'train':
                loader = train_loader
            else:
                loader = val_loader

            for i, (samples, labels) in enumerate(loader):
                samples = samples.to(device)
                labels = labels.to(device)
                samples = samples.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.LongTensor)
                
                
                #forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(samples)['out']
                    predictions = torch.argmax(outputs, 1)
                    loss = criterion(outputs, labels)

                    #backwards
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    #accumulates losses per epoch
                    train_losses[epoch] += loss.item()
                    running_loss += loss.item()
                    result = (predictions == labels)
                    running_correct += torch.sum(result)

                    
            if phase == 'train':
                epoch_loss = running_loss / len(train_loader)
                loss_writer.add_scalar("training loss", epoch_loss, epoch + 1)
                epoch_acc = running_correct.double() / (len(train_loader) * image_size * image_size * 10)
                acc_writer.add_scalar("training accuracy", epoch_acc, epoch + 1)
            else:
                epoch_loss = running_loss / len(val_loader)
                loss_writer.add_scalar("validation loss", epoch_loss, epoch + 1)
                epoch_acc = running_correct.double() / (len(val_loader) * image_size * image_size)
                acc_writer.add_scalar("validation accuracy", epoch_acc, epoch + 1)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            #deep copy the model
            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_weights)
                torch.save(model.state_dict(), PATH)

    
    print(f'Training with lr={learning_rate} and num_epochs={num_epochs} complete')

def train_model(model, learning_rate, num_epochs, weights):
    batch_size = 5
    train_set_labels = []
    train_set_samples = []

    data_loading.loadDatasets('/content/drive/MyDrive/Python/labels/train_set.txt', train_set_labels)
    data_loading.loadDatasets('/content/drive/MyDrive/Python/samples/train_set.txt', train_set_samples)
    train_dataset = PascalVOC.PascalVOCDataset(train_set_samples, train_set_labels, transforms.train_transform, transforms.target_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    print("Training model. This will take a moment...")
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=21)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #training loop
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        train_losses = [0 for i in range(num_epochs)]
        epochs = []
        epochs.append(epoch)

        running_loss = 0.0
        running_correct = 0
        for i, (samples, labels) in enumerate(train_loader):
            
            samples = samples.to(device)
            labels = labels.to(device)
            samples = samples.type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.LongTensor)

            outputs = model(samples)['out']
            predictions = torch.argmax(outputs, 1)
            
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses[epoch] += loss.item()
            running_loss += loss.item()
            result = (predictions == labels)
            running_correct += torch.sum(result)

            if i == 1:
                print(torch.numel(predictions))
                # print("Labels:", labels)
                # print("Outputs:", outputs)
                # print("Predictions:", predictions)
                print("Number of correct pixels:", torch.sum(result))
                torch.save(model.state_dict(), PATH)

        epoch_loss = running_loss / len(train_loader)
        loss_writer.add_scalar("training loss", epoch_loss, epoch + 1)
        epoch_acc = running_correct.double() / (batch_size * image_size * image_size * len(train_loader))
        acc_writer.add_scalar("training accuracy", epoch_acc, epoch + 1)
        

    print(f'Training with lr={learning_rate} and num_epochs={num_epochs} complete')