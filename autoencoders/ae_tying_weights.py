# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:44:09 2019

@author: Jean Mário
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial
import sys
import os

#Unfortunately, implementing tied weights in TensorFlow using the fully_connec
#ted() function is a bit cumbersome; it’s actually easier to just define the layers manually.
#The code ends up significantly more verbose:

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0005

activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

weights1_init = initializer([n_inputs,n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])

#weights
weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.transpose(weights2, name="weights3") #tied weights
weights4 = tf.transpose(weights1, name="weights4") #tied weights

#bieases




