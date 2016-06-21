#!/usr/bin/env python
# ==========================================================================================
#
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
#
# License: see the LICENSE.TXT
# ==========================================================================================


#from __future__ import absolute_import
#from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils

import os
import numpy as np

import sct_utils as sct
from msct_image import Image


def load_data(path_data, label_value, verbose=1):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    """
    from sys import stdout
    if verbose == 1:
        sct.printv('Extracting ' + path_data)
    cr = '\r'

    list_data = []
    for root, dirs, files in os.walk(path_data):
        for file in files:
            if file.endswith('.nii.gz') and file.find('_seg') == -1:
                list_data.append(root + '/' + file)

    nx, ny, nz, nt, px, py, pz, pt = Image(list_data[0]).dim
    size = (nx, ny)

    num_data = len(list_data)
    data = np.empty((num_data, 1, nx, ny), dtype="float32")
    label = np.zeros(num_data, dtype="uint8")

    if list_data is None:
        for i, fname_im in enumerate(list_data):
            if verbose == 1:
                stdout.write(cr)
                stdout.write(str(i) + '/' + str(num_data))
            im_data = Image(path_data + fname_im)
            data_image = np.asarray(im_data.data, dtype="float32")
            data_image = (data_image - np.min(data_image)) / (np.max(data_image) - np.min(data_image))  # output between 0 and 1
            data[i, :, :] = data_image[:, :]
            label[i] = label_value

    return data, label, size

# Load data
data_in, label_in, size_in = load_data(path_data='/Users/benjamindeleener/data/machine_learning/train/in/', label_value=1)
print 'Number of samples IN =', data_in.shape[0]
data_out, label_out, size_out = load_data(path_data='/Users/benjamindeleener/data/machine_learning/train/out/', label_value=0)
print 'Number of samples IN =', data_out.shape[0]
data = np.concatenate((data_in, data_out), axis=0)
label = np.concatenate((label_in, label_out), axis=0)
print 'Total number of samples =', data.shape[0]

# label 0 to 9 of 10 categories, keras requested format is binary class matrices, transforming it, directly call this function keras provided
label = np_utils.to_categorical(label, 2)

batch_size = 32
nb_epoch = 1

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, 50, 50)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2, input_dim=512))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.fit(data, label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, show_accuracy=True)

#score = model.evaluate(X_test, y_test, batch_size=16)
