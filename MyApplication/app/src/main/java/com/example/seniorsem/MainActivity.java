package com.example.seniorsem;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;

import org.pytorch.Module;

import java.io.FileNotFoundException;

public class MainActivity extends AppCompatActivity {

    ImageView imageview;
    Bitmap bitmap = null;
    Module module = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageview = findViewById(R.id.imageView);
    }

    public void loadImage(View view){
        int STORAGE_PERMISSION_CODE = 23;
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, STORAGE_PERMISSION_CODE);
        Intent intent = new Intent (Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, 0);
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode == RESULT_OK){
            Uri targetUri = data.getData();
            try{
                bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(targetUri));
                imageview.setImageBitmap(bitmap);
            } catch (FileNotFoundException e){
                e.printStackTrace();
            }
        }
    }

    /*public void getSegmentation(View view){

    }*/



}