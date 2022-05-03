# Overview
This will serve as our group repository for our senior sem class. We will train our model and use it in an android application that loads an image and segments the image.
# Cloning the repo
`git clone https://github.com/Humberto1945/CapStoneProject.git`
# Application Start
1. Open android studio and navagate to the `CapStoneProject\MyApplication` and verify the application builds correctly.
2. Download a model we trained [here](https://drive.google.com/drive/folders/1Xu78SN1wcKR4UbMBWs5h70WVoRUyk6tp).
3. Put the model in CapStoneProject\MyApplication\app\src\main
4. Verify you have python downloaded by typing python and it should return "Python 3.10.2" or whatever version of python you have.
5. Open up the model_loading.py and change `FILE =` to the directory that you put your model in.
6. On line 15 you can change the name of the file. Example change /model3 to /model4. Keep the rest of the directory the same.`CapStoneProject/MyApplication/app/src/main/assets/model3.pth`
7. Next add an assests directory to the main folder allowing the outputted model to be placed there.
8. Navigate in the terminal to CapStoneProject\MyApplication\app\src\main and run the command `python model_loading.py`
9. Verify the model outputed in the `assests folder`
10. Also change the asset name to what you named it above `preTrainedModel= Module.load(readingFilePath(this,"model3.pth"));`
11. If you are running on a virtual device make sure you have the `Pixel 5 API 30` as your virtual device. 
12. If you are running on a physical device make sure you have developer mode on and plug in your phone to the pc.

# Running the application
Select the green trangle button to run the application

![image](https://user-images.githubusercontent.com/60196726/165429929-2048ca90-7898-4599-8566-1c7694f827c1.png)

It might take a few seconds to load the application, but when it does the app will open automatically.
![image](https://user-images.githubusercontent.com/60196726/165430319-1bbb256b-1ca9-4243-b38b-95e6bc726715.png)

Finally, click `LOAD` to select an image from the gallery and then select `SEGMENT` to see the image segmented.

# Model Training
Navigate to the Python folder in the project and locate the .py files that are used for model training and testing.
The `models.py` file contains the dataloading and model training sections of the project. The file uses the PascalVOC dataset images to train with. These images are available [here](https://drive.google.com/drive/folders/18jOfvgxQKa2vJVjoJeQHJmMmZDigJx6f?usp=sharing) on this Google Drive link. It is designed so that the files can be used inside Google Colab. The shared drive must be dragged and dropped into "My Drive" so that the files cn be accessed when mounting the drive onto Google Colab. Then, the cell where models.py is located can be edited to specify a name to save the model under. This model will be saved directly into the drive under the models/ subdirectory. This is the location where the other models which were trained during the project were saved. Before training, change the name of the directory where the loss and accuracy logs for tensorboard will be stored, so as to not save two logs under the same directory.

`loss_writer = SummaryWriter("/content/drive/MyDrive/Python/logs_deeplab/small_lr_loss")`

`acc_writer = SummaryWriter("/content/drive/MyDrive/Python/logs_deeplab/small_lr_acc")`

The model's hyperparameters can be changed in the last line of `models.py` by editing `learning_rate` and `num_epochs`.

`train_model(resnet50deeplab, learning_rate=0.00001, num_epochs=200)`

# Class Weights
The file `class_weights.py` can be run to retrieve the class weights for the imbalanced frequency classes of the PascalVOC Dataset. This file requires that the drive with the dataset is mounted so that it can compute the frequencies for each class in the training dataset. These values are used as a tenor and are placed inside the `nn.CrossEntropyLoss()` function as a parameter. It allows the model to train with these weights and acheive more balanced accuracies per class.

# Model Testing
The file `accuracy_testing.py` is used to compute the accuracy of the trained model using the Intersection over Union metric. Before this is run on Colab, the drive must be mounted and `!pip install torchmetrics` must be run in a cell to use the `torchmetrics.JaccardIndex()` function which computes the IoU. The FILE variable must be updated to reflect which model to compute the IoU for.

`FILE = "/content/drive/MyDrive/Python/models/resnet50_good_lr.pth"`

The last line of the code specifies whether the Deeplab or FCN models should be loaded to be compatible with the model that is in the FILE path. It should be:

`test_accuracy(resnet50fcn)`

or

`test_accuracy(resnet50deeplab)`

# Normalization
The file `MeanSDCalc.py` calculates the mean and standard deiation of the dataset per channel and these values are used in the transforms to normalize the pixel values. It uses the PascalVOC dataset in the drive to calculate these values. This file can be run to retrieve the values that are used in the training and testing loop in the following section:

`train_transform = transforms.Compose([`

  `   transforms.ToPILImage(),`
  
  `   transforms.CenterCrop(image_size),`
  
  `   transforms.ToTensor(),`
  
  `transforms.Normalize([0.457, 0.443, 0.407], [0.273, 0.269, 0.285])`
  
`])`


# Creating Datasets
The file `createDatasets.py` can be run to create text files which hold the image file names for test, training, and validation sets in text files. These text files are used in the dataloading sections of the training and testing loops to load the data into the model.

# Helpful Notes When cloning repository
When you git clone our project and open it with android studio make sure you open on the level "MyApplication" to make sure it builds correctly.

