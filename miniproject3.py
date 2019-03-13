# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:28:21 2019

@author: Sarah
"""

import pickle
import pandas as pd
#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
    

class number_analysis:
    def __init__(self):
        #training folder must be in the same directory as this script
        self.negative_folder = "./train/neg/"
        self.positive_folder = "./train/pos/"
    
    def load_training_data(self):
        
        #gets training input
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

############################################################################################################

    
#Initialize the CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit_generator(x_training, steps_per_epoch = 8000, epochs = 10, validation_data = y_train, validation_steps = 800)