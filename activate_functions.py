#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:03:45 2019

@author: jean
"""

import numpy as np

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
    if (sum >= 0):
        return sum
    return 0

#Linear Function - Utilizada par exemplos de regressão
def linear(sum):
    return sum

#Softmax 
    #Utilizada para retornar probabilidades
    #Apliucável a problemas com mais de duas classes para classificação
def softmax(x):
    ex = np.exp(x)
    return ex/ex.sum()

