# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:31:36 2019

@author: Jean Mário
"""

from rbm import RBM
import numpy as np

#num_visible -> nos visiveis: ENTRADAS
#num_hidden -> n neuronios na camada oculta
rbm = RBM(num_visible = 6, num_hidden = 2)

base = np.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,1],
                 [0,0,1,1,0,1],
                 [0,0,1,1,0,1]])

rbm.train(base, max_epochs=5000)

#primeira linha é bias
#o restante das linhas é os filmes
#primeiro item de cada coluna também é bias
#Para 3 filmes de terror -> positivo, 
rbm.weights

user1 = np.array([[1,1,0,1,0,0]])
user2 = np.array([[0,0,0,1,1,0]])

rbm.run_visible(user2)

camada_escondida = np.array([[1,0]])
recomendacao = rbm.run_hidden(camada_escondida)