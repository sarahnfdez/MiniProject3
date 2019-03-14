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

from sklearn.preprocessing import MinMaxScaler


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
    
    def preprocess(self, X, y):
        import numpy as np
        from scipy import ndimage
        import matplotlib.pyplot as plt
        
        np.random.seed(3)
        X = x
        resized_x = np.zeros([40000, 40, 40])
        resized_y = np.zeros([40000])
        j = 0;
        i = 0;
        for im in X[0:80]:
            print(i)
   
          #  print(i)
            n = 64
            l = 64
            points = l*np.random.random((2, n**2))
            im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
          
            mask = im > im.mean() + im.std()
            
            
            label_im, nb_labels = ndimage.label(mask)
            
            # Find the largest connected component
            sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
            maximum = np.max(sizes)
            mask_size = sizes < 100
            remove_pixel = mask_size[label_im]
            label_im[remove_pixel] = 0

            labels = np.unique(label_im)
            label_im = np.searchsorted(labels, label_im)
            labels = np.unique(label_im)
            
            
            # Now that we have only one connected component, extract it's bounding box
            max_area=0
            x_length = 0
            y_length = 0
            final_x_slice = slice(0, 0)
            final_y_slice = slice(0, 0)
            largest_y_slice_x = slice(0, 0)
            largest_y_slice_y = slice(0, 0)
            largest_x_slice_x = slice(0, 0)
            largest_x_slice_y = slice(0, 0)
            mpl.pyplot.imshow(label_im)
            for label in labels:
                
                if label==0:
                    continue
                slice_x, slice_y = ndimage.find_objects(label_im==label)[0]
                
                x_slice = slice_x.stop - slice_x.start
                y_slice = slice_y.stop - slice_y.start
                area = x_slice*y_slice
                if (x_slice> x_length):
                    x_length = x_slice
                    largest_x_slice_x = slice_x
                    largest_x_slice_y = slice_y
                if (y_slice > y_length):
                    y_length = y_slice
                    largest_y_slice_x = slice_x
                    largest_y_slice_y = slice_y
            if y_length > x_length:
                slice_x = largest_y_slice_x
                slice_y = largest_y_slice_y
            else:
                slice_x = largest_x_slice_x
                slice_y = largest_x_slice_y
                    
                        
            final_x_slice = slice_x
            final_y_slice = slice_y
                
            slice_x = slice(final_x_slice.start, final_x_slice.stop)
            slice_y = slice(final_y_slice.start, final_y_slice.stop)
                
            roi = label_im[slice_x, slice_y]
            
            x_pad_size_left = ((40 - roi.shape[1])/2)
            x_pad_size_right = (40 - roi.shape[1])/2
            y_pad_size_left = ((40 - roi.shape[0])/2)
            y_pad_size_right = ((40 - roi.shape[0])/2)
            if np.floor(x_pad_size_left) != x_pad_size_left:
                x_pad_size_left = (np.floor(x_pad_size_left))
                x_pad_size_right = (x_pad_size_left + 1)
            if  np.floor(y_pad_size_left) != y_pad_size_left:
                y_pad_size_left = ( np.floor(y_pad_size_left))
                y_pad_size_right = ( y_pad_size_left + 1   )  
            try:
                roi = np.pad(roi, [(int(y_pad_size_left), int(y_pad_size_right)), (int(x_pad_size_left), int(x_pad_size_right))], 'constant', constant_values=(0,0))
            except:
                i+=1
                continue
                
            roi = np.reshape(roi, [1, 40, 40])

            resized_x[j] = roi

            resized_y[j] = y[i]
            i+=1
            j+=1
        return resized_x, resized_y

        

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

x_train, y_train = CNN_analysis.preprocess(x, x, y)

w=15
h=15
fig=mpl.pyplot.figure(figsize=(8, 8))
columns = 5
rows = 16
for i in range(len(x_train)):
    img = x_train[i]
    fig.add_subplot(rows, columns, i+1)
    mpl.pyplot.imshow(img)
plt.show()
#scaler = StandardScaler()
#show = x[17]
mpl.pyplot.imshow(x[54])
#test1.run_CNN1(x_train, y_train)


