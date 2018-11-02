#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from skimage.transform import rescale
from sklearn.utils import shuffle
from keras.models import load_model

paddingSize = 128000
depthIntervalFrame = 8

def import_depth_data(action, subject, trial):
    filename = f'Depth/a{action}_s{subject}_t{trial}_depth.mat'

    if Path(filename).is_file():
        mat = sio.loadmat(filename)
        return mat['d_depth']
    else:
        return None
    
def add_padding (image_array):
    size = len(image_array [0][0])
    features = []
    
    for i in range(0,size,depthIntervalFrame):
        img = image_array[40:,80:240,i]
        feature_image = rescale(img, 1.0/2.0)
        
        features_process = feature_image.flatten()
        features = np.append (features, features_process)
        
    print ('Depth feature size =', len (features))
    features = np.pad(features, (0, paddingSize - len(features)), 'constant')
    
    return features


action = 1
subject = 1
trial = 1
counter = 0
numberAction = 0

total = 0
y = []

test = np.empty([paddingSize, ], dtype=np.float64)

while action < 28:
    subject = 1
    numberAction = 0
    
    while subject < 9:
        trial = 1
        while trial < 5:
            try:
                content = import_depth_data(action,subject,trial)
                content = add_padding(content)
                
                test = np.vstack((test, content))
                
                trial += 1
                counter += 1
                numberAction += 1
                
            except:
                trial += 1
            
        subject += 1
        
    labellist = [action] * numberAction
    y += labellist   
    action += 1
    
test = np.delete(test,0,0)

print ("finish")

scaled_X, y = shuffle(test, y, random_state=256)

x_train, x_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state = 2)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


itemList = []
item = {}

model = Sequential()
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(28, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=128, batch_size=32, validation_split=0.1)

# Test the model

scores = model.evaluate(x_test, y_test)
print(scores[1]*100)
print ("\nAccuracy: %.2f%%" % (scores[1]*100))

plt.figure()
plt.title('Skeleton Model - Validation Loss', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])

plt.figure()
plt.title('Skeleton Model - Validation Accuracy', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Training', 'Validation'], loc='lower right')

model.summary()


model.save('depth_rescale_model.h5')
del model

model = load_model('depth_rescale_model.h5')