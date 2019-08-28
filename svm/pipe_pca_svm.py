# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 18:52:06 2019

@author: Jean Mário
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

#### FRESADORA

def fresadora(x):
    return (2/5) + ((1/10) * np.exp(-5*x)) - ((1/2)*np.exp(-x))

#####

#load data
X = np.arange(-30,30,0.05,dtype='float32')
Y = np.sin(X) + 2 * np.cos(X) 
#Y = np.power(X,2)
X = X.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#model
svr = SVR(kernel='rbf',C=10000,gamma=0.13,epsilon=0.04)
pca = PCA(n_components=1)
svr_pca = Pipeline([('reduce_dim',pca), ('svr',svr)])


svr.fit(X_train,y_train)
svr_pca.fit(X_train,y_train)

y_pred_SVRPCA = svr_pca.predict(X_test)
y_pred = svr.predict(X_test)

print("MSE e MAE: ")
print(metrics.mean_squared_error(y_test,y_pred_SVRPCA))
print(metrics.mean_absolute_error(y_test,y_pred_SVRPCA))
print(metrics.mean_squared_error(y_test,y_pred))
print(metrics.mean_absolute_error(y_test,y_pred))

print("Variance score: ")
print(metrics.explained_variance_score(y_test,y_pred_SVRPCA))
print(metrics.explained_variance_score(y_test,y_pred))

#PLOTTING
svrs = [svr_pca, svr]
kernel_label = ['SVR with PCA', 'SVR']
model_color = ['m','c']

fig, axs = plt.subplots(nrows = 2, ncols = 2)
axs[0,0].plot(y_test,color='g')
axs[0,0].plot(y_pred_SVRPCA,'b:')
axs[0,1].plot(y_pred_SVRPCA-y_test,color='red')
axs[0,1].set_xlabel("erro")
axs[1,0].plot(y_test,color='g')
axs[1,0].plot(y_pred,'b:')
axs[1,1].plot(y_pred-y_test,color='red')
plt.show()