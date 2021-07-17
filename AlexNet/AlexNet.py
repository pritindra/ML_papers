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
from sklearn.model_selection import train_test_split
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
y_one_hot = OneHotEncoder(sparse= False).fit_transform(y_comb.reshape(len(y_comb),1))

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size = 0.2, random_state=1)

"""
# convulational layer - 5
def cn_layer(fx,y = False):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(fx,(3,3), activation="relu"))
    if (y == True):
        model.add(tf.keras.layers.BatchNormalization())
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
    model.add(tf.keras.layers.Input((40,20,1,)))

    model.add(cn_layer(f1, True))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), padding = "same"))
    model.add(cn_layer(f2,True))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), padding = "same"))
    model.add(cn_layer(f3))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), padding = "same"))

    model.add(tf.keras.layers.Flatten())
    model.add(fc_layer(h1))
    model.add(fc_layer(h2))
    model.add(tf.keras.layers.Dense(19, activation = "softmax")) # 19 according to labels
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

"""

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input((40,20,1,)))

model.add(tf.keras.layers.Conv2D(96,(11,11),activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding="same"))

model.add(tf.keras.layers.Conv2D(256,(5,5),activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding="same"))

model.add(tf.keras.layers.Conv2D(384,(3,3),activation="relu"))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(384,(3,3),activation="relu"))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(256,(3,3),activation="relu"))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2048,activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(2048,activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(19, activation="softmax"))

traingen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=5, width_shift_range=[-2,2])
traingen.fit(X_train)
model.summary()

reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=20, verbose=1)
checkp = tf.keras.callbacks.ModelCheckpoint('./result_model.h5', monitor = 'val_loss', verbose = 1, save_best_only = True)

history = model.fit(traingen.flow(X_train, y_train, batch_size = 32), validation_data = (X_test, y_test), epochs = 150, steps_per_epoch = len(X_train)/32, callbacks = [checkp])



