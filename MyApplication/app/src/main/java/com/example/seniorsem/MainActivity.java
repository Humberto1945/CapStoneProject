package com.example.seniorsem;

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
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.FileNotFoundException;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    Uri selectedImage;
    Bitmap imageBitmap;
    // Module moduleResNet;
    int imageSize = 224;
    // class segmentation corresponding indices
    int background = 0;
    int unlabelled = 225;
    int aeroplane = 1;
    int bicycle = 2;
    int bird = 3;
    int boat = 5;
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
    int monitor = 20;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // Setting buttons
        Button loadImage = findViewById(R.id.loadButton);
        Button segmentImage = findViewById(R.id.segmentButton);



        // Lets the user select an image from their camera roll
        loadImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, 50);
            }
        });
        // Action of clicking the segment button
        segmentImage.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(imageBitmap,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB);
                // Getting the input values
                final float[] inputs = inputTensor.getDataAsFloatArray();
                // TODO extract data from the inputs based on the model
                int width = imageBitmap.getWidth();
                int height = imageBitmap.getHeight();


                int[] intValues = new int[width * height];
                // for loop to go through outputted size

                for(int i = 0; i< width; i++){
                    for(int j = 0; j< height; j++ ){

                    }
                }
                // Displaying the segmented images
                ImageView imageView = findViewById(R.id.imageView);
                Bitmap segmentedBitMap = Bitmap.createScaledBitmap(imageBitmap, width, height, true);
                // Copying the segmented bit map into an output to translate into pixels
                Bitmap outputBitmap = segmentedBitMap.copy(segmentedBitMap.getConfig(), true);
                outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
                imageView.setImageBitmap(outputBitmap);

            }
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
        ImageView imageView = findViewById(R.id.imageView);
        selectedImage = data.getData();
        imageView.setImageURI(selectedImage);
        try {
            //Getting the bitmap image
            imageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),selectedImage);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /*public void getSegmentation(View view){

    }*/



}