#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:10:33 2019

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

# Imports csv into pandas DataFrame object.
Eckerle4_df = pandas.read_csv("db/Eckerle4.csv", header=0)
 
# Converts dataframes into numpy objects.
Eckerle4_dataset = Eckerle4_df.values.astype("float32")
# Slicing all rows, second column...
X = Eckerle4_dataset[:,1]
# Slicing all rows, first column...
y = Eckerle4_dataset[:,0]
 
# Data Scaling from 0 to 1, X and y originally have very different scales.
X_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_scaled = ( X_scaler.fit_transform(X.reshape(-1,1)))
y_scaled = (y_scaler.fit_transform(y.reshape(-1,1)))
 
# Preparing test and train data: 60% training, 40% testing.
X_train, X_test, y_train, y_test = train_test_split( \
    X_scaled, y_scaled, test_size=0.40, random_state=3)


model = Sequential()
 
# Input layer with dimension 1 and hidden layer i with 128 neurons. 
model.add(Dense(128, input_dim=1, activation='relu'))
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
model.fit(X_train, y_train, nb_epoch=256, batch_size=2, verbose=2, validation_split=0.1)

predicted = model.predict(X_test)
 
# Plot in blue color the predicted adata and in green color the
# actual data to verify visually the accuracy of the model.
pyplot.plot(y_scaler.inverse_transform(predicted), color="blue")
pyplot.plot(y_scaler.inverse_transform(y_test), color="green")
pyplot.show()

