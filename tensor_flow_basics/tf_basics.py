# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:49:34 2019

@author: Jean Mário
"""


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

#1
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

#2
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()


#3    
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
    
#A TensorFlow program is typically split into two parts: the first part builds a computation
#graph (this is called the construction phase), and the second part runs it (this is
#the execution phase). The construction phase typically builds a computation graph
#representing the ML model and the computations required to train it. The execution
#phase generally runs a loop that evaluates a training step repeatedly (for example, one
#step per mini-batch), gradually improving the model parameters
    
tf.reset_default_graph()

#Lifecycle of a node value

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval()) # 10
    print(z.eval()) # 15

with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val) # 10
    print(z_val) # 15  


#Linear regression with TF
#Normal Equation (θ = XT · X)–1 · XT
    
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()    