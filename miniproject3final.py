# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:13:53 2019
@author: Sarah
"""

from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation, GlobalAveragePooling2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras import layers
import pickle
import pandas as pd
from scipy import stats
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
#import matplotlib as mpl
import numpy as np
from scipy import ndimage
from random import random, seed
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image 


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
    
    def load_test_data(self):
        dataset = open('test_images.pkl', 'rb')
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
        np.random.seed(3)
        resized_x = np.zeros([X.shape[0], 40, 40])
        resized_y = np.zeros([X.shape[0]])
        j = 0;
        i = 0;
        for im in X:
         #   print(i)
          #  im = X[1]
          #  print(i)
            n = 64
            l = 64
            #points = l*np.random.random((2, n**2))
            im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
          
            mask = im > im.mean() + im.std()
            
            
            label_im, nb_labels = ndimage.label(mask)
            
            # Find the largest connected component
            sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
            #maximum = np.max(sizes)
            mask_size = sizes < 100
            remove_pixel = mask_size[label_im]
            label_im[remove_pixel] = 0

            labels = np.unique(label_im)
            label_im = np.searchsorted(labels, label_im)
            labels = np.unique(label_im)
            
            
            # Now that we have only one connected component, extract it's bounding box
            #max_area=0
            x_length = 0
            y_length = 0
            final_x_slice = slice(0, 0)
            final_y_slice = slice(0, 0)
            largest_y_slice_x = slice(0, 0)
            largest_y_slice_y = slice(0, 0)
            largest_x_slice_x = slice(0, 0)
            largest_x_slice_y = slice(0, 0)
#            mpl.pyplot.imshow(label_im)
            for label in labels:
                
                if label==0:
                    continue
                slice_x, slice_y = ndimage.find_objects(label_im==label)[0]
                
                x_slice = slice_x.stop - slice_x.start
                y_slice = slice_y.stop - slice_y.start
                #area = x_slice*y_slice
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
                roi = roi.astype('uint8')
                roi = Image.fromarray(roi)
                roi = roi.resize((32,32))
                roi = np.array(list(roi.getdata()))
                roi = np.reshape(roi, [32,32])
                roi = np.pad(roi, [(4,4), (4,4)], 'constant', constant_values=(0,0))
                
            roi = np.reshape(roi, [1, 40, 40])

            resized_x[j] = roi

            resized_y[j] = y[i]
            i+=1
            j+=1
         #   mpl.pyplot.imshow(resized_x[79])
         
        resized_x = resized_x.reshape(resized_x.shape[0], 40, 40, 1)
        resized_x = resized_x.astype('float32')
        return resized_x, resized_y

        

    def make_babyNet1(self):
        
        input_shape = (40, 40, 1)

        #X_training /= 255
    
        # Creating a Sequential Model and adding the layers
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3,3), input_shape=input_shape))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(3,3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        model.add(Dense(64, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10,activation=tf.nn.softmax))

        model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

        return model
    
    def make_babyNet2(self):
        
        model = Sequential()
        input_shape = (40, 40, 1)
        LEARN_RATE = 1.0e-4
    
        model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=input_shape))    
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same'))  
        model.add(BatchNormalization())
        model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same', strides = 2))    
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Conv2D(192, kernel_size=(3, 3), activation='relu', padding = 'same'))    
        model.add(Conv2D(192, kernel_size=(3, 3), activation='relu', padding = 'same'))
        model.add(BatchNormalization())
        model.add(Conv2D(192, kernel_size=(3, 3), activation='relu', padding = 'same', strides = 2))    
        model.add(Dropout(0.5))    
    
        model.add(Conv2D(192, kernel_size=(3, 3), activation='relu', padding = 'same'))
        model.add(BatchNormalization())
        model.add(Conv2D(192, kernel_size=(1, 1), activation='relu', padding='valid'))
        model.add(Conv2D(10, kernel_size=(1, 1), activation='relu', padding='valid'))
        model.add(Dense(64, activation=tf.nn.relu))
        model.add(Dense(10,activation=tf.nn.softmax))

        model.add(GlobalAveragePooling2D())
        
        model.compile(loss='categorical_crossentropy', # Better loss function for neural networks
              optimizer=Adam(lr=LEARN_RATE), # Adam optimizer with 1.0e-4 learning rate
              metrics = ['accuracy']) # Metrics to be evaluated by the model
        
        return model
        
    def make_babyNet3(self):
        
        # instantiate model
        model = Sequential()

        # we can think of this chunk as the input layer
        model.add(Dense(64, input_dim=14, init='uniform'))
        model.add(Activation('tanh'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # we can think of this chunk as the hidden layer    
        model.add(Dense(64, init='uniform'))
        model.add(Activation('tanh'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # we can think of this chunk as the output layer
        model.add(Dense(2, init='uniform'))
        model.add(Activation('softmax'))
        model.add(BatchNormalization())
        
        # setting up the optimization of our weights 
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd)
        
        return model
    
    def make_babyNet4(self):
        input_shape = (40, 40, 1)
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3,3), input_shape=input_shape))
        model.add(Dense(64, activation=tf.nn.relu))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3,3), input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3,3), input_shape=input_shape))
        model.add(Dense(10,activation=tf.nn.softmax))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        return model        
        
    def make_LeNet(self):
        input_shape = (40, 40, 1)
        model = Sequential()

        model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.AveragePooling2D())
        
        model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(layers.AveragePooling2D())
        
        model.add(layers.Flatten())
        
        model.add(layers.Dense(units=120, activation='relu'))
        
        model.add(layers.Dense(units=84, activation='relu'))
        
        model.add(layers.Dense(units=10, activation = 'softmax'))
        
                
        model.compile(loss='sparse_categorical_crossentropy', # Better loss function for neural networks
              optimizer=Adam(lr=0.1), # Adam optimizer with 1.0e-4 learning rate
              metrics = ['accuracy']) 
        
        return model
    
    def sample(self, x_train, y_train, num_samples):
        seed(1)
        max_val = len(y_train)
        
        x = np.zeros([num_samples, 40, 40, 1])
        y = np.zeros([num_samples])
        counter = 0
        for i in range(num_samples):
            rand_num = random();
            rand_index = int(np.floor(rand_num * max_val))
            x[counter] = x_train[rand_index]
            y[counter] = y_train[rand_index]
            counter += 1
        return x, y
        
#####################################################################################################################
  
      
        
test1= CNN_analysis()
x, y = test1.load_training_data()
x_1, y_1 = test1.load_test_data()


x_train, y_train = test1.preprocess(x, y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
x_test, y_test = CNN_analysis.preprocess(x_1, x_1, y_1)


model_list = []
val_acc_list= []
num_samples = 30000
## do some bagging: subsample 40 tims.
for i in range(15):
    x_subsample, y_subsample = test1.sample(x_train, y_train, num_samples)
    model = test1.make_babyNet1()
    earlystopping = EarlyStopping(monitor='val_acc', patience=1)
    callbacks_list = [earlystopping]
    history = model.fit(x_subsample, y_subsample, epochs = 100,  validation_split=0.2, callbacks=callbacks_list)
    model_list.append(model)
    val_acc_list.append(history.history['val_acc'][-1])

y_pred = np.zeros([y_test.shape[0], 0 ])
for model in model_list:
    pred = model.predict_classes(x_test)
    pred = np.reshape(pred, [pred.shape[0], 1])
    y_pred = np.append(y_pred, pred, axis=1)
final_prediction = stats.mode(y_pred.T)
final_prediction = final_prediction[0]

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_val, final_prediction.T)


new_mod = model_list[0]
pred = new_mod.predict_classes(x_train)
model = test1.make_babyNet1()
checkpoint = ModelCheckpoint("checkpoint_model", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystopping = EarlyStopping(monitor='val_acc', patience=3)
callbacks_list = [checkpoint, earlystopping]


model.fit(x_train, y_train, epochs = 100,  validation_split=0.2, callbacks=callbacks_list)


y_pred = model.predict_classes(x_test)




import pandas as pd
y_pred = pd.DataFrame(final_prediction.T)
y_pred.to_csv("bagging.csv")
plt.imshow(x_test[1])
