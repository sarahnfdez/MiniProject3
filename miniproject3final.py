# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:13:53 2019

@author: Sarah
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.datasets import mnist
import pickle
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

class CNN_analysis:
    def __init__(self):
        return None
        
    def load_training_data(self):
        dataset = open('train_images.pkl', 'rb')
        X_training = pickle.load(dataset)
            
        #gets training y
        y_training = []
        y = []
        df = pd.read_csv("train_labels.csv")
        y_training = df.values.tolist()
        for arr in y_training:
            element = arr[1]
            y.append(element)
        
        return X_training, y
    
    def run_CNN1(self, X_training, y):
       
        X_training = X_training.reshape(X_training.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)
        
        # Making sure that the values are float so that we can get decimal points after division
        X_training = X_training.astype('float32')
        
        # Normalizing the RGB codes by dividing it to the max RGB value.
        X_training /= 255
    
        # Creating a Sequential Model and adding the layers
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(3,3), input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3,3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10,activation=tf.nn.softmax))

        model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

        model.fit(x=X_training, y=y, epochs=10, validation_split = 0.2)
        
#####################################################################################################################
  
      
        
test1= CNN_analysis()
x, y = test1.load_training_data()
scaler = StandardScaler()
show = scaler.fit_transform(x[0])
mpl.pyplot.imshow(show)
test1.run_CNN1(x_train, y_train)


import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

np.random.seed(3)
n = 64
l = 64
im = x[11020]
points = l*np.random.random((2, n**2))
im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
plt.imshow(im)
mask = im > im.mean()+im.std()*1.5
plt.imshow(mask)

label_im, nb_labels = ndimage.label(mask)

# Find the largest connected component
sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
maximum = np.max(sizes)
mask_size = sizes < maximum
remove_pixel = mask_size[label_im]
label_im[remove_pixel] = 0
labels = np.unique(label_im)
label_im = np.searchsorted(labels, label_im)
labels = np.unique(label_im)


# Now that we have only one connected component, extract it's bounding box
max_area=0
final_x_slice = slice(0, 0)
final_y_slice = slice(0, 0)
for label in labels:

    if label==0:
        continue
    slice_x, slice_y = ndimage.find_objects(label_im==label)[0]
    
    x_slice = slice_x.stop - slice_x.start
    y_slice = slice_y.stop - slice_y.start
    area = x_slice*y_slice
    if (area> max_area):
        max_area = area
        final_x_slice = slice_x
        final_y_slice = slice_y
    
slice_x = final_x_slice
slice_y = final_y_slice
    
roi = im[slice_x, slice_y]

plt.figure(figsize=(4, 2))
plt.axes([0, 0, 1, 1])
plt.imshow(roi)
plt.axis('off')


