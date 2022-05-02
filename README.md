# Overview
This will serve as our group repository for our senior sem class. We will train our a model and use it in an android application that loads an image and segments the image.
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


# Helpful Notes When cloning repository
When you git clone our project and open it with android studio make sure you open on the level "MyApplication" to make sure it builds correctly.
