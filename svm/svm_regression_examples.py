# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:12:01 2019

@author: Jean Mário
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import SVR

#load data

X = np.arange(-32,32,0.05,dtype='float32')

Y = np.power(X,2)
X = X.reshape(-1,1)

#Normalization and preprocessing
#norm = MinMaxScaler(feature_range=(0,1))
#X = norm.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

svr_rbf = SVR(kernel='rbf', C=100,gamma=0.01,epsilon=.1)
svr_rbf.fit(X_train,y_train)

y_pred = svr_rbf.predict(X_test)

metrics.mean_squared_error(y_test,y_pred)
metrics.mean_absolute_error(y_test,y_pred)








