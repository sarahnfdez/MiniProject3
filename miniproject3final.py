# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:13:53 2019

@author: Sarah
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import pickle
import pandas as pd
import tensorflow as tf

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
        
        X_training = X_training.reshape(X_training.shape[0], 64, 64, 1)
        input_shape = (64, 64, 1)
        
        # Making sure that the values are float so that we can get decimal points after division
        X_training = X_training.astype('float32')
        
        # Normalizing the RGB codes by dividing it to the max RGB value.
        X_training /= 255
    
        # Creating a Sequential Model and adding the layers
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
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
test1.run_CNN1(x, y)
