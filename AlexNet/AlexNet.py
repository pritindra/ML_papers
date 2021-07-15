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
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import train_test_split
path = "./samples"

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
for image in os.listdir(path):
    if image[6:] != "png":
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

# print(X.shape)
# print(Y.shape)

X /= 255.0

y_comb = LabelEncoder().fit_transform(Y)
y_one_hot = OneHotEncoder(sparse= False).fit_transform(y_combine.reshape(len(y_combine),1))

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size = 0.2)

# convulational layer - 5
def cn_layer(fx,y):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(fx,(3,3), activation="relu"))
    if y = True:
        model.add(tf.keras.layers.BatchNormalization())
    else:
        continue
    model.add(tf.keras.layers.Dropout(0.2))
    return model    

def fc_layer(hx,y):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(hx, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    return model

def cnn(f1,f2,f3,h1,h2):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(40,20,1,))

    model.add(cn_layer(f1, True))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), padding = "same"))
    model.add(cn_layer(f2,True))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), padding = "same"))
    model.add(cn_layer(f3,False))
    model.add(cn_layer(f3,False))
    model.add(cn_layer(f2,False))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), padding = "same"))

    model.add(tf.keras.layers.Flatten())
    model.add(fc_layer(h1))
    model.add(fc_layer(h2))
    model.add(tf.keras.layers.Dense(19, activation = "softmax")) # 19 according to labels
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


