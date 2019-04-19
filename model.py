##############################################################
# Create training data.
##############################################################

import csv
import matplotlib.image as mpimg
import numpy as np

# Correction term for left and right camera steering angles.
CORRECTION = 0.3

# Load driving log data into memory.
lines = []
with open('./data/driving_log.csv') as csv_file:
  reader = csv.reader(csv_file)
  for line in reader:
    lines.append(line)
  lines = lines[1:]  # Delete header.

images = []
measurements = []
for row in lines:
  # Read in steering angle (for center camera).
  steering_center = float(row[3])

  # Create adjusted steering measurements for the side camera images.
  steering_left = np.clip(steering_center + CORRECTION, -1., 1.)
  steering_right = np.clip(steering_center - CORRECTION, -1., 1.)

  # Read in images from center, left and right cameras.
  imgs_dir = './data/'
  img_center = mpimg.imread(imgs_dir + row[0])
  img_left = mpimg.imread(imgs_dir + row[1][1:])  # Delete leading space.
  img_right = mpimg.imread(imgs_dir + row[2][1:])  # Delete leading space.
    
  # Flip images and steering angles.
  img_center_flip = np.fliplr(img_center)
  img_left_flip = np.fliplr(img_left)
  img_right_flip = np.fliplr(img_right)
  steering_center_flip = -steering_center
  steering_left_flip = -steering_left
  steering_right_flip = -steering_right

  # Add images and steering angles to data set.
  images.extend([img_center, img_left, img_right,
                 img_center_flip, img_left_flip, img_right_flip])
  measurements.extend([steering_center, steering_left, steering_right,
                       steering_center_flip, steering_left_flip, steering_right_flip])

# Final training/validation data.
X_train = np.array(images)
y_train = np.array(measurements)

##############################################################
# Train model.
##############################################################

from keras.models import Sequential
from keras.layers import Activation, Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 128
NB_EPOCH = 3

# Number of rows of pixels to crop off the top of the image.
TOP_CROP = 50
# Number of rows of pixels to crop off the bottom of the image.
BOTTOM_CROP = 20

model = Sequential()

# Preprocessing & normalization.
model.add(Cropping2D(cropping=((TOP_CROP, BOTTOM_CROP), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x/255.0) - 0.5))

# Define architecture.
model.add(Convolution2D(6, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(16, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(32, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

# Train the model.
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.2, nb_epoch=NB_EPOCH)

model.save('model.h5')
exit()