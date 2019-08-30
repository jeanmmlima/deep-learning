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
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import matplotlib.pyplot as plt

X = np.arange(-30,30,0.05,dtype='float32')
X = X.reshape(-1,1)
Y = 2 * X

### Normalization - Necessário para a Bernouli RBM

normalizer = MinMaxScaler(feature_range=(0,1))
X = normalizer.fit_transform(X)
Y = normalizer.fit_transform(Y)

###

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

rbm = BernoulliRBM(n_components=1, n_iter=400, learning_rate=0.35, verbose=True, batch_size=30)
rbm2 = BernoulliRBM(n_components=1, n_iter=200, learning_rate=0.4,verbose=True, batch_size=20)
svr = SVR(kernel='rbf',C=10000,gamma=0.3,epsilon=0.15)
rbm_svr = clone(svr)
rbm2_svr = clone(svr)
pipe_rbm_svr = Pipeline([(('rbm'),clone(rbm)),(('svr'),clone(svr))])


pipe_dbn = Pipeline([(('rbm1'),clone(rbm)),(('rbm2'),clone(rbm)),(('svr'),clone(svr))])


rbm.fit(X_train)
nX = rbm.transform(X_train)
nX_test = rbm.transform(X_test)
rbm_svr.fit(nX,y_train)

rbm2.fit(nX)
n2X = rbm2.transform(nX)
n2X_test = rbm2.transform(nX_test)
rbm2_svr.fit(n2X,y_train)

svr.fit(X_train,y_train)
pipe_rbm_svr.fit(X_train,y_train)
pipe_dbn.fit(X_train,y_train)

y_pred_rbmsvr = rbm_svr.predict(nX_test)
y_pred = svr.predict(X_test)
y_pred_pipe = pipe_rbm_svr.predict(X_test)
y_pred_dbn = pipe_dbn.predict(X_test)
y_pred_rbm2 = pipe_dbn.predict(n2X_test)

print("MSE: ")
print(metrics.mean_squared_error(y_test,y_pred_rbmsvr))
print(metrics.mean_squared_error(y_test,y_pred))
print(metrics.mean_squared_error(y_test,y_pred_pipe))
print(metrics.mean_squared_error(y_test,y_pred_dbn))
print(metrics.mean_squared_error(y_test,y_pred_rbm2))

print("Variance score: ")
print(metrics.explained_variance_score(y_test,y_pred_rbmsvr))
print(metrics.explained_variance_score(y_test,y_pred))
print(metrics.explained_variance_score(y_test,y_pred_pipe))
print(metrics.explained_variance_score(y_test,y_pred_dbn))
print(metrics.explained_variance_score(y_test,y_pred_rbm2))




