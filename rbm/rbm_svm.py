# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:46:11 2019

@author: Jean Mário
"""


import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

X = np.arange(-30,30,0.05,dtype='float32')
X = X.reshape(-1,1)
Y = 2*X

### Normalization - Necessário para a Bernouli RBM

normalizer = MinMaxScaler(feature_range=(0,1))
X = normalizer.fit_transform(X)
Y = normalizer.fit_transform(Y)

###

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

rbm = BernoulliRBM(n_components=1, n_iter=150, learning_rate=0.3, verbose=True)
rbm.fit(X_train)

nX = rbm.transform(X_train)
nX_test = rbm.transform(X_test)

rbm_svr = SVR(kernel='rbf',C=1000,gamma=0.3,epsilon=0.05)
rbm_svr.fit(nX,y_train)

svr = SVR(kernel='rbf',C=1000,gamma=0.3,epsilon=0.05)
svr.fit(X_train,y_train)

y_pred_rbmsvr = rbm_svr.predict(nX_test)
y_pred = svr.predict(X_test)

print("MSE e MAE: ")
print(metrics.mean_squared_error(y_test,y_pred_rbmsvr))
print(metrics.mean_absolute_error(y_test,y_pred_rbmsvr))
print(metrics.mean_squared_error(y_test,y_pred))
print(metrics.mean_absolute_error(y_test,y_pred))

print("Variance score: ")
print(metrics.explained_variance_score(y_test,y_pred_rbmsvr))
print(metrics.explained_variance_score(y_test,y_pred))



