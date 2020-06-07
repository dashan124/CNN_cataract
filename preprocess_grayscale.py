from tensorflow import keras
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import glob 
from os import listdir,makedirs
from os.path import isfile,join
from numpy import *
from PIL import Image
import cv2
import os 

path_1="/home/dashan/Desktop/augmented_data"
path_2="/home/dashan/Desktop/modified_data"

p1=path_1+"/1_normal"
path=p1
IMG_SIZE=300
#first 300 images of normal eye
# def process_normal_eye_images():
files = [f for f in listdir(path) if isfile(join(path,f))] 
# print(len(files))
# print('bc')
for image in files:
    img = cv2.imread(os.path.join(path,image))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray = cv2.resize(gray, (200,200)) 
    gr = cv2.resize(gray,(IMG_SIZE,IMG_SIZE))
    print("lol1")
    dstPath = join(path_2+"/1_normal",image)
    print("lol2")
    cv2.imwrite(dstPath,gr)
    print("lol3")
'''
then 100 images of cataract eye
'''
p2=path_1+"/2_cataract"
path=p2
files = [f for f in listdir(path) if isfile(join(path,f))] 
for image in files:
    img = cv2.imread(os.path.join(path,image))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gr = cv2.resize(gray,(IMG_SIZE,IMG_SIZE))
    dstPath = join(path_2+"/2_cataract",image)
    cv2.imwrite(dstPath,gr)
'''
then 101 images of glucoma eye
'''

# p3=path_1+"/2_glaucoma"
# path=p3
# files = [f for f in listdir(path) if isfile(join(path,f))] 
# for image in files:
#     img = cv2.imread(os.path.join(path,image))
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     gr = cv2.resize(gray,(IMG_SIZE,IMG_SIZE))
#     dstPath = join(path_2+"/2_glaucoma",image)
#     cv2.imwrite(dstPath,gr)
# '''
# 100 images of ratina disease
# '''

# p4=path_1+"/3_retina_disease"
# path=p4
# files = [f for f in listdir(path) if isfile(join(path,f))] 
# for image in files:
#     img = cv2.imread(os.path.join(path,image))
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     gr = cv2.resize(gray,(IMG_SIZE,IMG_SIZE))
#     dstPath = join(path_2+"/3_retina_disease",image)
#     cv2.imwrite(dstPath,gr)