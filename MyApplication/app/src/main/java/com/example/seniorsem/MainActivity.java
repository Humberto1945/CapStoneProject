package com.example.seniorsem;

<<<<<<< Updated upstream
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.pytorch.Module;

import java.io.FileNotFoundException;
=======
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
//import org.pytorch.LiteModuleLoader;

import java.io.File;
import java.io.FileOutputStream;
>>>>>>> Stashed changes
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    Uri selectedImage;
    Bitmap imageBitmap;
<<<<<<< Updated upstream
    // Module moduleResNet;
    int imageSize = 224;
=======
    Module preTrainedModel;
    ImageView imageView;
    Tensor inputTensor;
    Tensor outputTensor;
    int unlabelled = 21;
    int background = 0;
    int aeroplane = 1;
    int bicycle = 2;
    int bird = 3;
    int boat = 4;
    int bottle =5;
    int bus = 6;
    int car =7;
    int cat = 8;
    int chair = 9;
    int cow = 10;
    int diningtable =11;
    int dog = 12;
    int horse = 13;
    int mortorbike =14;
    int person = 15;
    int pottedPlant = 16;
    int sheep = 17;
    int sofa = 18;
    int train = 19;
    int tvmonitor = 20;
    int CLASSNUM = 21;

>>>>>>> Stashed changes

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // Setting buttons
        Button loadImage = findViewById(R.id.loadButton);
        Button segmentImage = findViewById(R.id.segmentButton);
<<<<<<< Updated upstream


        // Lets the user select an image from their camera roll
        loadImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, 50);
            }
=======
        // Import the pretrained-model
        try {
            preTrainedModel = Module.load(readingFilePath(this, "model.pt"));

        } catch (IOException e) {
            e.printStackTrace();
        }
        // Lets the user select an image from their camera roll
        loadImage.setOnClickListener(view -> {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(intent, 50);
        });


        // Action of clicking the segment button
        segmentImage.setOnClickListener(view -> {
            // getting the inputs from the image
            inputTensor = TensorImageUtils.bitmapToFloat32Tensor(imageBitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

            Map<String, IValue> outTensors = preTrainedModel.forward(IValue.from(inputTensor)).toDictStringKey();
            // VOC key word out to tensor
            outputTensor = outTensors.get("out").toTensor();
            float[] scores = outputTensor.getDataAsFloatArray();
            // getting the sizes of the image
            int width = imageBitmap.getWidth();
            int height = imageBitmap.getHeight();
            // find the initial values
            int[] intValues = new int[width * height];
            imageBitmap.getPixels(intValues, 0, imageBitmap.getWidth(), 0, 0, imageBitmap.getWidth(), imageBitmap.getHeight());
            // looping through to get the class that the model thinks the image is
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    int maxi = 0, maxj = 0, maxk = 0;
                    double maxnum = -Double.MAX_VALUE;

                    for (int i = 0; i < CLASSNUM; i++) {
                        float score = scores[i * (width * height) + j * width + k];
                        if (score > maxnum) {
                            maxnum = score;
                            maxi = i; maxj = j; maxk = k;
                        }
                    }
                    // receiving the output class number and assigning to the specified class
                    // also setting the rgb of the class object
                    if (maxi == background)
                        intValues[maxj * width + maxk] =  Color.rgb(0,0,0);
                    else if (maxi == aeroplane)//brick red
                        intValues[maxj * width + maxk] = Color.rgb(128,0,0);
                    else if (maxi == bicycle)//medium green
                        intValues[maxj * width + maxk] = Color.rgb(0,128,0);
                    else if (maxi == bird)//green yellow
                        intValues[maxj * width + maxk] = Color.rgb(128,128,0);
                    else if (maxi == boat)//dark blue
                        intValues[maxj * width + maxk] = Color.rgb(0,0,128);
                    else if (maxi == bottle)//purple
                        intValues[maxj * width + maxk] = Color.rgb(128,0,128);
                    else if (maxi == bus)//turquoise
                        intValues[maxj * width + maxk] = Color.rgb(0,128,128);
                    else if (maxi == car)//light grey
                        intValues[maxj * width + maxk] = Color.rgb(128,128,128);
                    else if (maxi == cat)//dark brown
                        intValues[maxj * width + maxk] = Color.rgb(64,0,0);
                    else if (maxi == chair)//red
                        intValues[maxj * width + maxk] = Color.rgb(192,0,0);
                    else if (maxi == cow)//green
                        intValues[maxj * width + maxk] = Color.rgb(64,128,0);
                    else if (maxi == diningtable)//mustard/gold
                        intValues[maxj * width + maxk] = Color.rgb(192,128,0);
                    else if (maxi == dog)//deep purple
                        intValues[maxj * width + maxk] = Color.rgb(64,0,128);
                    else if (maxi == horse)//magenta
                        intValues[maxj * width + maxk] = Color.rgb(192,0,128);
                    else if (maxi == mortorbike)//pale blue
                        intValues[maxj * width + maxk] = Color.rgb(64,128,128);
                    else if (maxi == person)//pale pink
                        intValues[maxj * width + maxk] = Color.rgb(192,128,128);
                    else if (maxi == pottedPlant)//dark green
                        intValues[maxj * width + maxk] = Color.rgb(0,64,0);
                    else if (maxi == sheep)//light brown
                        intValues[maxj * width + maxk] = Color.rgb(128,64,0);
                    else if (maxi == sofa)//light green
                        intValues[maxj * width + maxk] = Color.rgb(0,192,0);
                    else if (maxi == train)//neon yellow
                        intValues[maxj * width + maxk] = Color.rgb(128,192,0);
                    else if (maxi == tvmonitor )//blue
                        intValues[maxj * width + maxk] = Color.rgb(0,64,128);
                    else if (maxi == unlabelled )//off white
                        intValues[maxj * width + maxk] = Color.rgb(224,224,192);
                }
            }

            // rescaling the image for better input
            Bitmap bitmapScaled = Bitmap.createScaledBitmap(imageBitmap, width, height, true);
            // putting a copy of the bitmap inside an outputted variable
            Bitmap outputBitmap = bitmapScaled.copy(bitmapScaled.getConfig(), true);
            outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
            final Bitmap transferredBitmap = Bitmap.createScaledBitmap(outputBitmap, imageBitmap.getWidth(), imageBitmap.getHeight(), true);
            // sets the segmented image to the imageview
            ImageView imageView = findViewById(R.id.imageView);
            imageView.setImageBitmap(transferredBitmap);
>>>>>>> Stashed changes
        });
    }

    public void loadImage(View view){
        int STORAGE_PERMISSION_CODE = 23;
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, STORAGE_PERMISSION_CODE);
        Intent intent = new Intent (Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, 0);
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode, resultCode, data);
<<<<<<< Updated upstream
        ImageView imageView = findViewById(R.id.imageView);
        selectedImage = data.getData();
        imageView.setImageURI(selectedImage);
        try {
            //Getting the bitmap image
            imageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),selectedImage);
=======
        // checks to see if an image is actually selected
        // and if so sets the image in the image view
        if (resultCode == RESULT_OK && data != null) {
            selectedImage = data.getData();
            imageView = findViewById(R.id.imageView);
            imageView.setImageURI(selectedImage);
        }
        try {
            // getting the bitmap image
            imageBitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage));
            imageView.setImageBitmap(imageBitmap);
>>>>>>> Stashed changes
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /*public void getSegmentation(View view){

    }*/



}