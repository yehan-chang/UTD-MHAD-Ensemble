#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from random import randint
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from skimage.transform import rescale
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.utils import shuffle
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pandas_ml import ConfusionMatrix

# =============================================================================
# Global variable
# =============================================================================

paddingSizeRescale = 128000
paddingSizeHOG = 11520
depthIntervalFrame = 8

orient = 9  # HOG orientations
pix_per_cell = 30 # HOG pixels per cell default 8
cell_per_block = 2 # HOG cells per block
hog_feat = True # HOG features on or off

# =============================================================================
# Function definition for pre-processing
# =============================================================================

def import_inertial_data(action, subject, trial):
    filename = f'Inertial/a{action}_s{subject}_t{trial}_inertial.mat'

    if Path(filename).is_file():
        mat = sio.loadmat(filename)
        return mat['d_iner']
    else:
        return None
    
def import_depth_data(action, subject, trial):
    filename = f'Depth/a{action}_s{subject}_t{trial}_depth.mat'

    if Path(filename).is_file():
        mat = sio.loadmat(filename)
        return mat['d_depth']
    else:
        return None

def import_skeleton_data(action, subject, trial):
    filename = f'Skeleton/a{action}_s{subject}_t{trial}_skeleton.mat'

    if Path(filename).is_file():
        mat = sio.loadmat(filename)
        return mat['d_skel']
    else:
        return None

def add_padding_inertial (content):
    number = len(content)    
    content2 = content
    
    while number < 326:   
        content2 = np.vstack((content2, [0,0,0,0,0,0]))
        number += 1

    return content2 

def add_padding_skeleton (content):
       
    matrix = np.empty([125, ], dtype=np.float64)
    
    for i in range(20):
        for j in range(3):
            number = len(content[i][j]) 
            test001 = content[i][j].flatten('F')
            while number < 125:
                test001 = np.append (test001, 0)
                number += 1
            matrix = np.vstack((matrix, test001))

    matrix = np.delete(matrix,0,0)
    matrix = matrix.flatten('F')
    
    return matrix
    
def add_padding_rescale_depth (image_array):
    size = len(image_array [0][0])
    features = []
    
    for i in range(0,size,depthIntervalFrame):
        img = image_array[40:,80:240,i]
        feature_image = rescale(img, 1.0/2.0)
        
        features_process = feature_image.flatten()
        features = np.append (features, features_process)
        
    print ('Depth feature size =', len (features))
    features = np.pad(features, (0, paddingSizeRescale - len(features)), 'constant')
    
    return features

def add_padding_hog_depth (image_array):
    size = len(image_array [0][0])
    features = []
    
    for i in range(0,size,depthIntervalFrame):
        img = image_array[40:,80:240,i]
        hog_features, hog_image = get_hog_features(img, orient, 
                                                   pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        
        features = np.append (features, hog_features)
        
    print ('HOG feature size =', len (features))
    features = np.pad(features, (0, paddingSizeHOG - len(features)), 'constant')
    
    return features

# =============================================================================
# Define a function to return HOG features and visualization
# =============================================================================
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# =============================================================================
# Manually split data variable for training
# =============================================================================
x_test_inertial = np.empty([6, ], dtype=np.float64)
x_train_inertial = np.empty([6, ], dtype=np.float64)
y_test_inertial = []
y_train_inertial = []

x_test_skeleton = np.empty([7500, ], dtype=np.float64)
x_train_skeleton = np.empty([7500, ], dtype=np.float64)
x_test_rescale = np.empty([paddingSizeRescale, ], dtype=np.float64)
x_train_rescale = np.empty([paddingSizeRescale, ], dtype=np.float64)
x_test_hog = np.empty([paddingSizeHOG, ], dtype=np.float64)
x_train_hog = np.empty([paddingSizeHOG, ], dtype=np.float64)

y_test = []
y_train = []

action = 1
subject = 1
trial = 1
counter = 0

monitorList = []
monitoring_x_test = {}

# =============================================================================
# Reading of data
# =============================================================================
while action < 28:
    subject = 1
    numberAction_test_inertial = 0
    numberAction_train_inertial = 0
    numberAction_test = 0
    numberAction_train = 0
       
    while subject < 9:
        trial = 1
        while trial < 5:
            randomizer = randint(0, 9)
            
            try:
                if randomizer < 2:
                    monitoring_x_test = {}

                    content_inertial = import_inertial_data(action, subject, trial)
                    content_skeleton = add_padding_skeleton(import_skeleton_data(action, subject, trial))
                    content_depth = import_depth_data(action, subject, trial) 
                                 
                    x_test_inertial = np.vstack((x_test_inertial, content_inertial)).astype(np.float64)
                    numberAction_test_inertial += len(content_inertial)
                    
                    x_test_skeleton = np.vstack((x_test_skeleton, content_skeleton)).astype(np.float64)
                    
                    content_depth_rescale = add_padding_rescale_depth (content_depth)
                    content_depth_hog = add_padding_hog_depth (content_depth)
                    
                    x_test_rescale = np.vstack((x_test_rescale, content_depth_rescale))
                    x_test_hog = np.vstack((x_test_hog, content_depth_hog))
                    
                    numberAction_test += 1
                    
                    monitoring_x_test ['Action'] = action
                    monitoring_x_test ['Subject'] = subject
                    monitoring_x_test ['Trial'] = trial
                    monitoring_x_test ['FrameAction'] = len(content_inertial)
                    monitorList.append (monitoring_x_test)
                
                else:
                    content_inertial = import_inertial_data(action, subject, trial)
                    content_skeleton = add_padding_skeleton(import_skeleton_data(action, subject, trial))
                    content_depth = import_depth_data(action, subject, trial) 
                    
                    content_depth_rescale = add_padding_rescale_depth (content_depth)
                    content_depth_hog = add_padding_hog_depth (content_depth)
                                        
                    x_train_inertial = np.vstack((x_train_inertial, content_inertial)).astype(np.float64)
                    x_train_skeleton = np.vstack((x_train_skeleton, content_skeleton)).astype(np.float64)
                    
                    x_train_rescale = np.vstack((x_train_rescale, content_depth_rescale))
                    x_train_hog = np.vstack((x_train_hog, content_depth_hog))
                                        
                    numberAction_train_inertial += len(content_inertial)
                    numberAction_train += 1
                    
                trial += 1  
                
            except:
                trial += 1
            
        subject += 1

    labellist_test_inertial = [action] * numberAction_test_inertial    
    y_test_inertial += labellist_test_inertial
    
    labellist_train_inertial = [action] * numberAction_train_inertial   
    y_train_inertial += labellist_train_inertial
        
    labellist_test = [action] * numberAction_test
    y_test += labellist_test
    
    labellist_train = [action] * numberAction_train
    y_train += labellist_train
    
    action += 1
    
x_test_inertial = np.delete(x_test_inertial,0,0)
x_train_inertial = np.delete(x_train_inertial,0,0)
x_test_skeleton = np.delete(x_test_skeleton,0,0)
x_train_skeleton = np.delete(x_train_skeleton,0,0)
x_test_rescale = np.delete(x_test_rescale,0,0)
x_train_rescale = np.delete(x_train_rescale,0,0)
x_test_hog = np.delete(x_test_hog,0,0)
x_train_hog = np.delete(x_train_hog,0,0)

print ("Finish Data Processing")

# =============================================================================
# Start of Machine Learning
# =============================================================================
x_train_skeleton, y_train_skeleton = shuffle(x_train_skeleton, y_train, random_state=256)
x_train_rescale, y_train_rescale = shuffle(x_train_rescale, y_train, random_state=256)
x_train_hog, y_train_hog = shuffle(x_train_hog, y_train, random_state=256)

pca_model_rescale = PCA(n_components=0.98, svd_solver='full')
pca_model_hog = PCA(n_components=0.98, svd_solver='full')

x_train_rescale = pca_model_rescale.fit_transform(x_train_rescale)
x_train_hog = pca_model_hog.fit_transform(x_train_hog)

y_train_skeleton = to_categorical(y_train_skeleton)
y_train_rescale = to_categorical(y_train_rescale)
y_train_hog = to_categorical(y_train_hog)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train_inertial = to_categorical(y_train_inertial)
y_test_inertial = to_categorical(y_test_inertial)

# =============================================================================
# Model for Inertial Dataset
# =============================================================================
model_inertial = Sequential()

model_inertial.add(Dense(128,activation='relu'))
model_inertial.add(Dense(64, activation = 'relu'))
model_inertial.add(Dense(28, activation='softmax'))

model_inertial.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_inertial = model_inertial.fit(x_train_inertial, y_train_inertial, epochs=128, batch_size=32, validation_split=0.1) 

# =============================================================================
# Model for Skeleton Dataset
# =============================================================================
model_skeleton = Sequential()

model_skeleton.add(Dense(192, activation='relu'))
model_skeleton.add(Dense(176, activation='relu'))
model_skeleton.add(Dense(28, activation='softmax'))
   
model_skeleton.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_skeleton = model_skeleton.fit(x_train_skeleton, y_train_skeleton, epochs=256, batch_size=32, validation_split=0.1)
    
# =============================================================================
# Model for Depth Rescale Dataset (Include Dropout for regularization)
# =============================================================================
model_rescale = Sequential()

model_rescale.add(Dropout(0.2))
model_rescale.add(Dense(128, activation='relu'))
model_rescale.add(Dropout(0.2))
model_rescale.add(Dense(64, activation='relu'))
model_rescale.add(Dropout(0.2))
model_rescale.add(Dense(28, activation='softmax'))

model_rescale.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_rescale = model_rescale.fit(x_train_rescale, y_train_rescale, epochs=128, batch_size=32, validation_split=0.1)

# =============================================================================
# Model for Depth HOG
# =============================================================================
model = Sequential()

model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(28, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_hog = model.fit(x_train_hog, y_train_hog, epochs=128, batch_size=32, validation_split=0.1)
# =============================================================================
# Test Inertial model
# =============================================================================
# PREDICTED RESULTS
test_set = []
index = 0 
for item in monitorList:
    loopCount = item['FrameAction']
    count = 0
    toAdd = list()
    while count < loopCount:
        toAppend = list(x_test_inertial[index])
        toAdd.append(toAppend)
        count = count + 1
        index = index + 1
    test_set.append(np.array(toAdd))

y_results_set = []
for test_item in test_set:
    y_results = model_inertial.predict(test_item)
    y_results = y_results.argmax(axis=1)
    mode = Counter(y_results)
    mode = mode.most_common(1)
    mode = mode[0][0]
    y_results_set.append(mode)
    

# ACTUAL RESULTS
y_test_flat = y_test_inertial.argmax(axis=1)
y_actual_set = []
numTest = len(monitorList)
index = -1
for i in range(numTest):
    index = index + monitorList[i]['FrameAction']
    y_actual = y_test_flat[index]
    y_actual_set.append(y_actual)

y_results_set = np.array(y_results_set)
y_actual_set = np.array(y_actual_set)

c_matrix = confusion_matrix(y_actual_set, y_results_set)
c_matrix_inertial = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis]
acc_inertial = c_matrix_inertial.diagonal()
    
correct_result = 0
for i in range(27):
    correct_result = correct_result + c_matrix[i][i]

accuracy = correct_result / len(monitorList)

# =============================================================================
# Test Skeleton model
# =============================================================================
scores_skeleton = model_skeleton.evaluate(x_test_skeleton, y_test)
print (scores_skeleton[1]*100)
print ("\nAccuracy: %.2f%%" % (scores_skeleton[1]*100))

plt.figure()
plt.title('Skeleton Model - Validation Loss', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history_skeleton.history['loss'])
plt.plot(history_skeleton.history['val_loss'])
plt.legend(['Training', 'Validation'])

plt.figure()
plt.title('Skeleton Model - Validation Accuracy', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history_skeleton.history['acc'])
plt.plot(history_skeleton.history['val_acc'])
plt.legend(['Training', 'Validation'], loc='lower right')

y_pred_skel = model_skeleton.predict(x_test_skeleton).argmax(axis=1)
skel_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_skel)
#Now the normalize the diagonal entries
skel_matrix = skel_matrix.astype('float') / skel_matrix.sum(axis=1)[:, np.newaxis]
acc_skel = skel_matrix.diagonal()

# =============================================================================
# Test Depth Rescale model
# =============================================================================
x_test_rescale = pca_model_rescale.transform(x_test_rescale) 
scores_rescale = model_rescale.evaluate(x_test_rescale, y_test)
print (scores_rescale[1]*100)
print ("\nAccuracy: %.2f%%" % (scores_rescale[1]*100))

plt.figure()
plt.title('Depth Model - Validation Loss', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history_rescale.history['loss'])
plt.plot(history_rescale.history['val_loss'])
plt.legend(['Training', 'Validation'])

plt.figure()
plt.title('Depth Model - Validation Accuracy', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history_rescale.history['acc'])
plt.plot(history_rescale.history['val_acc'])
plt.legend(['Training', 'Validation'], loc='lower right')

y_pred_rescale = model_rescale.predict(x_test_rescale).argmax(axis=1)
rescale_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_rescale)
#Now the normalize the diagonal entries
rescale_matrix = rescale_matrix.astype('float') / rescale_matrix.sum(axis=1)[:, np.newaxis]
acc_rescale = rescale_matrix.diagonal()

# =============================================================================
# Test Depth HOG model
# =============================================================================
x_test_hog = pca_model_hog.transform(x_test_hog) 
scores_hog = model.evaluate(x_test_hog, y_test)
print (scores_hog[1]*100)
print ("\nAccuracy: %.2f%%" % (scores_hog[1]*100))

plt.figure()
plt.title('HOG Model - Validation Loss', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history_hog.history['loss'])
plt.plot(history_hog.history['val_loss'])
plt.legend(['Training', 'Validation'])

plt.figure()
plt.title('HOG Model - Validation Accuracy', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history_hog.history['acc'])
plt.plot(history_hog.history['val_acc'])
plt.legend(['Training', 'Validation'], loc='lower right')

y_pred_hog = model.predict(x_test_hog).argmax(axis=1)
hog_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_hog)
#Now the normalize the diagonal entries
hog_matrix = hog_matrix.astype('float') / hog_matrix.sum(axis=1)[:, np.newaxis]
acc_hog = hog_matrix.diagonal()

# =============================================================================
# Save the model
# =============================================================================
#model_inertial.save('Model/depth_inertial_model.h5')
#model_skeleton.save('Model/depth_skeleton_model.h5')
#model_rescale.save('Model/depth_rescale_model.h5')
#model.save('Model/depth_hog_model.h5')

# =============================================================================
# Start of Ensemble
# =============================================================================
result_iner = np.array(y_results_set)

skel_result = model_skeleton.predict(x_test_skeleton)
result_skel = skel_result.argmax(axis=1)

rescale_result = model_rescale.predict(x_test_rescale)
result_rescale = rescale_result.argmax(axis=1)

hog_result = model.predict(x_test_hog)
result_hog = hog_result.argmax(axis=1)

resultAllModel = np.vstack((result_iner, result_skel, result_rescale, result_hog)).T

accuracy_models = []
accuracy_models.append(accuracy)
accuracy_models.append(scores_skeleton[1])
accuracy_models.append(scores_rescale[1])
accuracy_models.append(scores_hog[1])

#Get the best model accuracy
best_index = accuracy_models.index(max(accuracy_models))

final_result = []
for item in resultAllModel:
    final_predict_set = Counter(item)
    final_predict = final_predict_set.most_common()
    if(len(final_predict) == 1):
        final_predict = final_predict[0][0]
    elif(final_predict[0][1] > final_predict[1][1]):
        final_predict = final_predict[0][0]
    else:
        final_predict = item[best_index]
    final_result.append(final_predict)
    
all_results_matrix = confusion_matrix(y_actual_set, np.array(final_result))
correct_result = 0
for i in range(27):
    correct_result = correct_result + all_results_matrix[i][i]

accuracy_ensemble = correct_result / len(monitorList)

target_names = ['Swipt Left', 'Swipe Right', 'Wave', 'Clap', 'Throw', 'Arm Cross', 'Basketball shoot', 'Draw X', 'Draw Circle CW', 'Draw Circle CCW', 'Draw Triangle', 'Bowling', 'Boxing', 'Baseball Swing', 'Tennis Swing', 'Arm Curl', 'Tennis Serve', 'Push', 'Knock', 'Catch', 'Pickup Throw', 'Jog', 'Walk', 'Sit to Stand', 'Stand to Sit', 'lunge', 'Squad']
print(classification_report(y_actual_set, np.array(final_result), target_names=target_names))

cm = ConfusionMatrix(y_actual_set, np.array(final_result))
cm.plot()
stats = cm.stats()
    
cmstats = dict(stats) 
cmstats2 = cmstats['class']  
cmstats2.to_csv('ensemble.csv', sep=',')


    
    
    
    
    
    
    
    
    
    
    
    
    
    
