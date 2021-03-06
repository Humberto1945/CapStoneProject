package com.example.seniorsem;
import androidx.appcompat.app.AppCompatActivity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {

    Uri selectedImage;
    Bitmap imageBitmap;
    Module preTrainedModel;
    int unlabelled = 225;
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


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // Setting buttons
        Button loadImage = findViewById(R.id.loadButton);
        Button segmentImage = findViewById(R.id.segmentButton);
        // Import the pretrained-model
        try {
            preTrainedModel= Module.load(readingFilePath(this,"model9.pth"));

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
            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(imageBitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
            // setting the values to a map to get the class num
            Map<String, IValue> outTensors = preTrainedModel.forward(IValue.from(inputTensor)).toDictStringKey();
            // VOC key word out to tensor
            final Tensor outputTensor = Objects.requireNonNull(outTensors.get("out")).toTensor();
            final float[] scores = outputTensor.getDataAsFloatArray();
            // getting the sizes of the image
            int width = imageBitmap.getWidth();
            int height = imageBitmap.getHeight();
            // find the initial values
            int[] intValues = new int[width * height];
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
                    if (maxi == aeroplane)
                        intValues[maxj * width + maxk] = Color.rgb(128,0,0);
                    else if (maxi == bicycle)
                        intValues[maxj * width + maxk] = Color.rgb(0,128,0);
                    else if (maxi == bird)
                        intValues[maxj * width + maxk] = Color.rgb(128,128,0);
                    else if (maxi == boat)
                        intValues[maxj * width + maxk] = Color.rgb(0,0,128);
                    else if (maxi == bottle)
                        intValues[maxj * width + maxk] = Color.rgb(128,0,128);
                    else if (maxi == bus)
                        intValues[maxj * width + maxk] = Color.rgb(0,128,128);
                    else if (maxi == car)
                        intValues[maxj * width + maxk] = Color.rgb(128,128,128);
                    else if (maxi == cat)
                        intValues[maxj * width + maxk] = Color.rgb(64,0,0);
                    else if (maxi == chair)
                        intValues[maxj * width + maxk] = Color.rgb(192,0,0);
                    else if (maxi == cow)
                        intValues[maxj * width + maxk] = Color.rgb(64,128,0);
                    else if (maxi == diningtable)
                        intValues[maxj * width + maxk] = Color.rgb(192,128,0);
                    else if (maxi == dog)
                        intValues[maxj * width + maxk] = Color.rgb(64,0,128);
                    else if (maxi == horse)
                        intValues[maxj * width + maxk] = Color.rgb(192,0,128);
                    else if (maxi == mortorbike)
                        intValues[maxj * width + maxk] = Color.rgb(64,128,128);
                    else if (maxi == person)
                        intValues[maxj * width + maxk] = Color.rgb(192,128,128);
                    else if (maxi == pottedPlant)
                        intValues[maxj * width + maxk] = Color.rgb(0,64,0);
                    else if (maxi == sheep)
                        intValues[maxj * width + maxk] = Color.rgb(128,64,0);
                    else if (maxi == sofa)
                        intValues[maxj * width + maxk] = Color.rgb(0,192,0);
                    else if (maxi == train)
                        intValues[maxj * width + maxk] = Color.rgb(128,192,0);
                    else if (maxi ==tvmonitor )
                        intValues[maxj * width + maxk] = Color.rgb(0,64,128);
                    else if (maxi ==unlabelled )
                        intValues[maxj * width + maxk] = Color.rgb(224,224,192);
                    else
                        intValues[maxj * width + maxk] = Color.rgb(0,0,0);
                }
            }

            // rescaling the image for better input
            Bitmap bitmapScaled = Bitmap.createScaledBitmap(imageBitmap, width, height, true);
            // putting a copy of the bitmap inside an outputted variable
            Bitmap outputBitmap = bitmapScaled.copy(bitmapScaled.getConfig(), true);
            outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
            // sets the segmented image to the imageview
            ImageView imageView = findViewById(R.id.imageView);
            imageView.setImageBitmap(outputBitmap);
        });

    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode, resultCode, data);
        // checks to see if an image is actually selected
        // and if so sets the image in the image view
        if (resultCode == RESULT_OK && data != null) {
            selectedImage = data.getData();
            ImageView imageView = findViewById(R.id.imageView);
            imageView.setImageURI(selectedImage);
        }
        try {
            // getting the bitmap image
            imageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),selectedImage);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }//end of onActivity

    // helper method to get the location of the model
    public static String readingFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
            if (file.exists() && file.length() > 0) {
                return file.getAbsolutePath();
            }

        try (InputStream is = context.getAssets().open(assetName)) {
         try (OutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();
        }
        return file.getAbsolutePath();
    }
}



}//end of activity