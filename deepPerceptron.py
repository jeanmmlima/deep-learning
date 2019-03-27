#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:55:43 2019

@author: jean
"""

# Numeric Python Library.
import numpy 
# Python Data Analysis Library.
import pandas
# Scikit-learn Machine Learning Python Library modules.
#   Preprocessing utilities.
from sklearn import preprocessing
#   Cross-validation utilities.
from sklearn.model_selection import train_test_split
# Python graphical library
from matplotlib import pyplot
 
# Keras perceptron neuron layer implementation.
from keras.layers import Dense
# Keras Dropout layer implementation.
from keras.layers import Dropout
# Keras Activation Function layer implementation.
from keras.layers import Activation
# Keras Model object.
from keras.models import Sequential

# Slicing all rows, second column...
X = np.arange(-np.pi/2, np.pi/2,0.01)
X = [X, X]
X = np.transpose(np.array(X))
# Slicing all rows, first column...
y = numpy.transpose(numpy.array(numpy.sin(X[:,0]) + numpy.cos(X[:,1])))
 
# Data Scaling from 0 to 1, X and y originally have very different scales.
X_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_scaled_1 = ( X_scaler.fit_transform(X[:,0].reshape(-1,1)))
X_scaled_2 = ( X_scaler.fit_transform(X[:,1].reshape(-1,1)))
y_scaled = (y_scaler.fit_transform(y.reshape(-1,1)))

components = range(0,len(X),1)
X_teste = numpy.zeros((315,2))


for i in components:
    X_teste[i,0] = X_scaled_1[i]
    X_teste[i,1] = X_scaled_1[i]

X_scaled = X_teste



X_train, X_test, y_train, y_test = train_test_split( \
    X_scaled, y_scaled, test_size=0.20, random_state=3)


model = Sequential()
 
# Input layer with dimension 2 and hidden layer i with 128 neurons. 
model.add(Dense(128, input_dim=2, activation='relu'))
# Dropout of 20% of the neurons and activation layer.
model.add(Dropout(.2))
model.add(Activation("linear"))
# Hidden layer j with 64 neurons plus activation layer.
model.add(Dense(64, activation='relu'))
model.add(Activation("linear"))
# Hidden layer k with 64 neurons.
model.add(Dense(64, activation='relu'))
# Output Layer.
model.add(Dense(1))
 
# Model is derived and compiled using mean square error as loss
# function, accuracy as metric and gradient descent optimizer.
model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
 
# Training model with train data. Fixed random seed:
numpy.random.seed(3)
model.fit(X_train, y_train, nb_epoch=256, batch_size=2, verbose=2, validation_split=0.15)

predicted = model.predict(X_test)
 
# Plot in blue color the predicted adata and in green color the
# actual data to verify visually the accuracy of the model.
pyplot.plot(y_scaler.inverse_transform(predicted), color="blue")
pyplot.plot(y_scaler.inverse_transform(y_test), color="green")
pyplot.show()

