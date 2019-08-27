#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:37:28 2019

@author: jeanmarioml
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

#carrega base
base = datasets.load_digits()

previsores = np.asarray(base.data,'float32')
classe = base.target

#normaliza base
normalizador = MinMaxScaler(feature_range=(0,1))
previsores = normalizador.fit_transform(previsores)

#divide conjuntos de treinamento e teste

previsores_trein, previsores_teste, classe_trein, classe_teste = train_test_split(previsores,classe,test_size=0.2, random_state=0)

#RBM
rbm = BernoulliRBM(random_state = 0)
rbm.n_iter = 25
#num de neuronios na camada oculta
rbm.n_components = 50
#algoritmo
naive_rbm = GaussianNB()

classificador_rbm = Pipeline(steps = [('rbm',rbm),('naive',naive_rbm)])
classificador_rbm.fit(previsores_trein,classe_trein)

plt.figure(figsize=(20,20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10,10,i+1)
    plt.imshow(comp.reshape((8,8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.show()    

previsores_rbm = classificador_rbm.predict(previsores_teste)

precisao_rbm = metrics.accuracy_score(previsores_rbm,classe_teste)

naive_simples = GaussianNB()
naive_simples.fit(previsores_trein,classe_trein)
previsores_naive = naive_simples.predict(previsores_teste)
precisao_naive = metrics.accuracy_score(previsores_naive,classe_teste)