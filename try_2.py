from tensorflow import keras
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import glob 
import numpy as np
from os import listdir,makedirs
from os.path import isfile,join
from numpy import *
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
import cv2
import os 

path_2="/home/dashan/Desktop/Major_project/modified_data"
local_path=path_2+"/1_normal"
files = [f for f in listdir(local_path) if isfile(join(local_path,f))]

random.shuffle(files) 
files=files[0:100]
for i in files:
	originalImage = cv2.imread(local_path+"/"+i)
	dstPath = join(path_2+"/training_data/1_normal",i)
	cv2.imwrite(dstPath,originalImage)
# print(len(files))
local_path=path_2+"/2_cataract"
files_1 = [f for f in listdir(local_path) if isfile(join(local_path,f))]
for i in files_1:
	files.append(i)
	originalImage = cv2.imread(local_path+"/"+i)
	dstPath = join(path_2+"/training_data/2_cataract",i)
	cv2.imwrite(dstPath,originalImage)

local_path=path_2+"/2_glaucoma"
files_2 = [f for f in listdir(local_path) if isfile(join(local_path,f))]
files_2=files_2[0:100]
for i in files_2:
	files.append(i)
	originalImage = cv2.imread(local_path+"/"+i)
	dstPath = join(path_2+"/training_data/2_glaucoma",i)
	cv2.imwrite(dstPath,originalImage)

local_path=path_2+"/3_retina_disease"
files_3 = [f for f in listdir(local_path) if isfile(join(local_path,f))]
for i in files_3:
	files.append(i)
	originalImage = cv2.imread(local_path+"/"+i)
	dstPath = join(path_2+"/training_data/3_retina_disease",i)
	cv2.imwrite(dstPath,originalImage)


batch_size=128
pic_size=10


IMAGE_SIZE = 10
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS = 20
BATCH_SIZE = 32
TEST_SIZE = 30

from keras.preprocessing.image import ImageDataGenerator




data_generator = ImageDataGenerator(rescale=1./255,
    validation_split=0.2)
# datagen_validation = ImageDataGenerator()
TRAINING_DIR=path_2+"/training_data"

train_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True, seed=13,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="training")

validation_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True, seed=13,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="validation")


print("data gen works#########")
print(train_generator.__getitem__(0))
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger
# from livelossplot.keras import PlotLossesCallback
# import efficientnet.keras as efn


model = Sequential()

input_shape = (10,10, 1)
model.add(Conv2D(32, (3, 3), border_mode='same', input_shape=input_shape, activation='relu'))
model.add(Conv2D(32, (3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
# model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
# model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
# model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()
model.compile(loss='categorical_crossentropy',
            optimizer=RMSprop(lr=0.1),
            metrics=['accuracy'])
# from keras.utils import to_categorical
import tensorflow as tf
from keras.utils.np_utils import to_categorical
# print("lol")
# tf.keras.utils.to_categorical(
#     train_generator, num_classes=4, dtype='float32'
# )
# print("bc ")
# input_data = input_data.reshape((-1, image_side1, image_side2, channels))
model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples //batch_size,
    epochs = 4)
print("idhar aaya kya")
model.save_weights('4_epochs.h5') 