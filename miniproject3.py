import pickle
import pandas as pd
import tensorflow as tf


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
    
X_training.shape

# Reshaping the array to 4-dims so that it can work with the Keras API
X_training = X_training.reshape(X_training.shape[0], 64, 64, 1)
input_shape = (64, 64, 1)
# Making sure that the values are float so that we can get decimal points after division
X_training = X_training.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_training /= 255
print('x_train shape:', X_training.shape)
print('Number of images in x_train', X_training.shape[0])


# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
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

model.fit(x=X_training,y=y, epochs=10)