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

#load data
X = np.arange(-32,32,0.05,dtype='float32')
Y = X + X
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

metrics.mean_squared_error(y_pred_SVRPCA,y_test)
metrics.mean_squared_error(y_pred,y_test)

#PLOTTING
svrs = [svr_pca, svr]
kernel_label = ['SVR with PCA', 'SVR']
model_color = ['m','c']

fig
plt.plot(y_pred_SVRPCA,color='blue',label='ERRO')
plt.plot(y_pred,color='red')
fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()