################################################Imports#######################################
import os
import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Cropping2D,Dropout
from keras.layers.convolutional import Conv2D
from sklearn.model_selection import train_test_split
import tensorflow as tf
#######################################Using code found in lesson 17.Generators#####################################################
#######################################Export Datat found in cvs file into sample arr#####################################################
Filepath = 'Training_Data/driving_log.csv'
samples = []
with open(Filepath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
#######################################Split training data into train and validation sets#####################################################
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

################use Generators as large amounts of data are processed#####################################################
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                #################################set steering value according to image side####################
                for i in range(3):
                    if i==0:
                        correction = 0 #center image
                    elif i ==1:
                        correction = 0.2 #left image
                    else:
                        correction = -0.2 #right image

                name = batch_sample[i].split('/')[-1]
                # print(name)

                ###########################append images and flipped version from the images in one array####################################
                center_image = cv2.imread(name)
                images.append(center_image)
                images.append(cv2.flip(center_image,1))
                ###########################append steering values coresponds to the images####################################
                center_angle = float(batch_sample[3])
                center_angle = center_angle + correction
                angles.append(center_angle)
                angles.append(center_angle*-0.1)
        # trim image to only see section with road
        X_train = np.array(images)
        y_train = np.array(angles)
        yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = Sequential()
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24,(5,5),strides = (2,2), activation = "relu"))
model.add(Dropout(0.5))
model.add(Conv2D(36,(5,5),strides = (2,2), activation = "relu"))

model.add(Conv2D(48,(5,5),strides = (2,2), activation = "relu"))

model.add(Conv2D(64,(3,3),activation = "relu"))

model.add(Conv2D(64,(3,3),activation = "relu"))

model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

with tf.device('/device:GPU:2'):
    history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])