import torch

def change_pixels(image, image_size, prediction):
    i = 0
    
    for j in range(image_size-1):
        for k in range(image_size-1):
            if prediction[i][j][k].item() == 0: #background
                # print("This is a background pixel")
                image[i][0][j][k] = 0
                image[i][1][j][k] = 0
                image[i][2][j][k] = 0
            elif prediction[i][j][k].item() == 1: #airplane
                image[i][0][j][k] = 128
                image[i][1][j][k] = 0
                image[i][2][j][k] = 0
            elif prediction[i][j][k].item() == 2: #bicycle
                image[i][0][j][k] = 0
                image[i][1][j][k] = 128
                image[i][2][j][k] = 0
            elif prediction[i][j][k].item() == 3: #bird
                image[i][0][j][k] = 128
                image[i][1][j][k] = 128
                image[i][2][j][k] = 0
            elif prediction[i][j][k].item() == 4: #boat
                image[i][0][j][k] = 0
                image[i][1][j][k] = 0
                image[i][2][j][k] = 128
            elif prediction[i][j][k].item() == 5: #bottle
                image[i][0][j][k] = 128
                image[i][1][j][k] = 0
                image[i][2][j][k] = 128
            elif prediction[i][j][k].item() == 6: #bus
                image[i][0][j][k] = 0
                image[i][1][j][k] = 128
                image[i][2][j][k] = 128
            elif prediction[i][j][k].item() == 7: #car
                image[i][0][j][k] = 128
                image[i][1][j][k] = 128
                image[i][2][j][k] = 128
            elif prediction[i][j][k].item() == 8: #cat
                image[i][0][j][k] = 192
                image[i][1][j][k] = 0
                image[i][2][j][k] = 0
            elif prediction[i][j][k].item() == 9: #chair
                image[i][0][j][k] = 64
                image[i][1][j][k] = 0
                image[i][2][j][k] = 0
            elif prediction[i][j][k].item() == 10: #cow
                image[i][0][j][k] = 192
                image[i][1][j][k] = 128
                image[i][2][j][k] = 0
            elif prediction[i][j][k].item() == 11: #diningtable
                image[i][0][j][k] = 64
                image[i][1][j][k] = 128
                image[i][2][j][k] = 0
            elif prediction[i][j][k].item() == 12: #dog
                image[i][0][j][k] = 192
                image[i][1][j][k] = 0
                image[i][2][j][k] = 128
            elif prediction[i][j][k].item() == 13: #horse
                image[i][0][j][k] = 64
                image[i][1][j][k] = 0
                image[i][2][j][k] = 128
            elif prediction[i][j][k].item() == 14: #motorbike
                image[i][0][j][k] = 192
                image[i][1][j][k] = 128
                image[i][2][j][k] = 128
            elif prediction[i][j][k].item() == 15: #person
                image[i][0][j][k] = 64
                image[i][1][j][k] = 128
                image[i][2][j][k] = 128
            elif prediction[i][j][k].item() == 16: #pottedplant
                image[i][0][j][k] = 0
                image[i][1][j][k] = 192
                image[i][2][j][k] = 0
            elif prediction[i][j][k].item() == 17: #sheep
                image[i][0][j][k] = 128
                image[i][1][j][k] = 192
                image[i][2][j][k] = 0
            elif prediction[i][j][k].item() == 18: #sofa
                image[i][0][j][k] = 0
                image[i][1][j][k] = 64
                image[i][2][j][k] = 0
            elif prediction[i][j][k].item() == 19: #train
                image[i][0][j][k] = 128
                image[i][1][j][k] = 64
                image[i][2][j][k] = 0
            elif prediction[i][j][k].item() == 20: #tvmonitor
                image[i][0][j][k] = 0
                image[i][1][j][k] = 192
                image[i][2][j][k] = 128
            elif prediction[i][j][k].item() == 21: #void/unlabelled
                image[i][0][j][k] = 224
                image[i][1][j][k] = 224
                image[i][2][j][k] = 192
    return image