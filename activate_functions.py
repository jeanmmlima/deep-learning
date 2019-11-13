#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:03:45 2019

@author: jean
"""

import numpy as np
import matplotlib.pyplot as plt

def step(sum):
    if (sum >= 1):
        return 1
    return 0

#Problemas de classificação binária
def sigmoide(sum):
    return 1/(1 + np.exp(-sum))

#Classificação
def tanh(sum):
    return ((np.exp(sum)-np.exp(-sum))/(np.exp(sum) + np.exp(-sum)))

#ReLU - Rectifier Linear Units
    # Utilizada em redes neurais convolucionais e profundas
    #Tende a ter melhores resultados com essa funcao de ativacao
def relu(sum):
    return np.maximum(0,sum)

#Linear Function - Utilizada par exemplos de regressão
def linear(sum):
    return sum

#Softmax 
    #Utilizada para retornar probabilidades
    #Apliucável a problemas com mais de duas classes para classificação
def softmax(x):
    ex = np.exp(x)
    return ex/ex.sum()



#dados
x = np.arange(-3,3,0.01)    

y_tanh = tanh(x)
y_sig = sigmoide(x)
y_relu= relu(x)


fig = plt.figure()
plt.plot(x,y_tanh,label='tanh',linewidth=2,linestyle='-.')
plt.plot(x,y_sig,label='sigmoide',linewidth=2, linestyle='--')
plt.plot(x,y_relu,label='ReLU',linewidth=2)
plt.legend()
plt.grid()
plt.title("Funções de Ativação")

plt.show()

fig.savefig('af.eps')
fig.savefig('af.png')







