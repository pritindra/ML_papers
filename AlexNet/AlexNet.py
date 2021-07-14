"""
Implementing AlexNet CNN architecture in a dataset of 5 letter
captcha images .
Dataset api :
kaggle datasets download -d fournierp/captcha-version-2-images

"""
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import os
import PIL from Image

path = "/samples"

# Adaptive Thresh holding images
def TH_img(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)    

# Closing images
def C_img(img):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,2), np.uint(8)))

# Dilating images
def D_img(img):
    return cv2.dilate(img, np.ones((2,2), np.uint(8)), iterations = 1)

# Smoothing images
def Sm_img(img):
    return cv2.GaussianBlur(img, (1,1), 0)

# appending to all images
X = []
Y = []
for image in os.listdir(path)
    if image[6:] != "png"
        continue
    
    img = cv2.imread(os.path.join(path,image), cv2.IMREAD_GRAYSCALE)
    img = TH_img(img)
    img = C_img(img)
    img = D_img(img)
    img = Sm_img(img)
    
    img_list = [img[10:50, 30:50], img[10:50, 50:70], img[10:50, 70:90], img[10:50, 90:110], img[10:50, 110:130]]
     
    for i in range(5):
        X.append(tf.keras.preprocessing.image.img_to_array(Image.fromarray(img_list[i])))
        Y.append(image[i])

X = np.array(X)
Y = np.array(Y)



