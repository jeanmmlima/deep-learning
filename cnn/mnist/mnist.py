#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:13:41 2019

@author: jeanmarioml
"""

import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPool2D


(X_trein, y_trein), (X_test,y_test) =mnist.load_data()

plt.imshow(X_trein[0],cmap='gray')
plt.title('Classe '+str(y_trein[0]))

previsores_trein = X_trein.reshape(X_trein.shape[0],28,28,1)
previsores_test = X_test.reshape(X_test.shape[0],28,28,1)

previsores_trein = previsores_trein.astype('float32')
previsores_test = previsores_test.astype('float32')

#normalização entre 0 e 1

previsores_trein /= 255
previsores_test /= 255

#cod da saida

classe_trein = np_utils.to_categorical(y_trein,10)