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
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


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
classe_test = np_utils.to_categorical(y_test,10)

classifier = Sequential()

#CONVOLUCAO
#Conv2D(num_filtros, tam_kernels, formato_entrada, func_ativacao),
#recomendavel 64 kernels como filtros (128,256,...)
classifier.add(Conv2D(32,(3,3), input_shape=(28,28,1), activation='relu'))

#EXTRA1: BATCH NORMALIZATION
classifier.add(BatchNormalization())

#POOLING
#percorre caracteristicas em 2 por 2
classifier.add(MaxPooling2D(pool_size =(2,2)))

#FLATTENING - nenhum parametro
classifier.add(Flatten())

#REDE NEURAL DENSA
#Dense(num_neuros, activation_func, )
classifier.add(Dense(units=128, activation='relu'))
#Saida - 10 saidas com softmaz
classifier.add(Dense(units = 10,activation='softmax'))

classifier.compile(loss = 'categorical_crossentropy',
                   optimizer='adam',metrics=['accuracy'])

classifier.fit(previsores_trein,classe_trein, 
               batch_size=128,epochs=5,
               validation_data=(previsores_test,classe_test))

result = classifier.evaluate(previsores_test,classe_test)