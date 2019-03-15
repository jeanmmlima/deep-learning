#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:58:49 2019

@author: jean
"""
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 5
np.random.seed(seed)

(X,y),(X_trein, y_trein) = mnist.load_data()

previsores = X.reshape(X.shape[0],28,28,1)
previsores = previsores.astype('float32')

previsores /= 255

classe = np_utils.to_categorical(y,10)

#controla validação cruzada
kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=seed)
results = []

b = np.zeros(shape=(classe.shape[0], 1))

for indice_treinamento, indice_teste in kfold.split(previsores, 
                                                    np.zeros(shape=(classe.shape[0], 1))):
    #print('Indices treinamento: ',indice_treinamento, 'Indice teste', indice_teste)
    classificador = Sequential()
    classificador.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
    classificador.add(MaxPooling2D(pool_size=(2,2)))
    classificador.add(Flatten())
    classificador.add(Dense(units=128,activation='relu'))
    classificador.add(Dense(units=10,activation='softmax'))
    classificador.compile(loss='categorical_crossentropy',optimizer='adam',
                          metrics=['accuracy'])
    classificador.fit(previsores[indice_treinamento],classe[indice_treinamento],
                      batch_size=128,epochs=5)
    precisao = classificador.evaluate(previsores[indice_teste],classe[indice_teste])
    results.append(precisao[1])
    

#media = results.mean()
media = sum(results)/len(results)