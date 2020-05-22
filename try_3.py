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

IMG_SIZE=300

def label_img(name):
	if name[0:2]=="NL":
		return np.array([1,0,0,0])
	elif name[0:2]=="ca":
		return np.array([0,1,0,0])
	elif name[0:2]=="Gl":
		return np.array([0,0,1,0])
	else:
		return np.array([0,0,0,1])
def load_training_data(arg):
	train_data=[]
	DIR=path_2+"/"+arg
	for img in os.listdir(DIR):
		label=label_img(img)
		# print(img)
		# continue
		path=os.path.join(DIR,img)
		img=Image.open(path)
		img=img.convert('L')
		img.resize((IMG_SIZE,IMG_SIZE),Image.ANTIALIAS)
		train_data.append([np.array(img),label])
		flip_img=Image.open(path)
		flip_img=flip_img.convert('L')
		flip_img = flip_img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
		flip_img = np.array(flip_img)
		flip_img = np.fliplr(flip_img)
		train_data.append([flip_img, label])
	shuffle(train_data)
	return train_data
a=load_training_data("1_normal")
b=load_training_data("2_cataract")
c=load_training_data("2_glaucoma")
d=load_training_data("3_retina_disease")
print(len(a),len(b),len(c),len(d))

train_data=[]
for i in a[0:200]:
	train_data.append(i)
for i in b[0:200]:
	train_data.append(i)
for i in c[0:200]:
	train_data.append(i)
for i in d[0:200]:
	train_data.append(i)

# IMG_SIZE=
trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
trainLabels = np.array([i[1] for i in train_data])
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization


model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
print("compile to ho gya ")

model.fit(trainImages, trainLabels, batch_size = 50, epochs = 5, verbose = 1)