# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:45:51 2019

@author: Jean Mário
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def fresadora(x):
    return (2/5) + ((1/10) * np.exp(-5*x)) - ((1/2)*np.exp(-x))

#####

#load data
X = np.arange(-30,30,0.05,dtype='float32')
#A = np.arange(30,-30,-0.05,dtype='float32')
#B = np.arange(-300,300,0.5,dtype='float32')
Y = np.sin(X) + 2 * np.cos(X)
#Y = np.power(X,2)
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)

#normaliza base
normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)
Y = normalizador.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#model

rbm = BernoulliRBM(n_iter=50000, n_components=1, random_state=0)
X_train_rbm = rbm.fit_transform(X_train)

#svr_rbm = Pipeline([('rbm',rbm), ('svr',svr)])
rbm_svr = SVR(kernel='rbf',C=1000,gamma=0.3,epsilon=0.05)
raw_svr = SVR(kernel='rbf',C=1000,gamma=0.3,epsilon=0.05)

rbm_svr.fit(X_train_rbm,y_train)
raw_svr.fit(X_train,y_train)


y_pred_SVRRBM = rbm_svr.predict(X_test)
y_pred = raw_svr.predict(X_test)

print("MSE e MAE: ")
print(metrics.mean_squared_error(y_test,y_pred_SVRRBM))
print(metrics.mean_absolute_error(y_test,y_pred_SVRRBM))
print(metrics.mean_squared_error(y_test,y_pred))
print(metrics.mean_absolute_error(y_test,y_pred))

print("Variance score: ")
print(metrics.explained_variance_score(y_test,y_pred_SVRRBM))
print(metrics.explained_variance_score(y_test,y_pred))

fig, axs = plt.subplots(nrows = 2, ncols = 2)
axs[0,0].plot(y_test,color='g')
axs[0,0].plot(y_pred_SVRPCA,'b:')
axs[0,1].plot(y_pred_SVRPCA-y_test,color='red')
axs[0,1].set_xlabel("erro")
axs[1,0].plot(y_test,color='g')
axs[1,0].plot(y_pred,'b:')
axs[1,1].plot(y_pred-y_test,color='red')
plt.show()