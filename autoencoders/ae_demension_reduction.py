# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:18:20 2019

@author: Jean Mário
"""

#Undercomplete AutoEncoders (AE)  for demensionality reduction (i.e 3D to 2D)

#if AE uses only linear activations and the cost function is the MSE,
# it can se shown that it ents up performing a PCA

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy.random as rnd
from sklearn.preprocessing import StandardScaler
import numpy as np
#matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

#DATA
rnd.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3))
data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)

scaler = StandardScaler()
X_train = scaler.fit_transform(data[:100])
X_test = scaler.transform(data[100:])

#MODEL
n_inputs = 3 #3D inputs
n_hidden = 2 #2D coding
n_outputs = n_inputs

learning_rate = 0.01

#Inserts a placeholder for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None,n_inputs])
hidden = fully_connected(X,n_hidden,activation_fn=None) #activation_fn=None -> performs PCA
outputs = fully_connected(hidden, n_outputs, activation_fn=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) #MSE for cost function

optimizer = tf.train.AdadeltaOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()


n_iterations = 1000
codings = hidden

#TRANING AND VALIDATIONS

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        training_op.run(feed_dict={X: X_train})
    codings_val = codings.eval(feed_dict={X: X_test})
    
#Plot
fig = plt.figure(figsize=(4,3))
plt.plot(codings_val[:,0], codings_val[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.show()
