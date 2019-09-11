# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:34:46 2019

@author: Jean Mário
"""

import numpy as np
import tensorflow as tf
import sys, os

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0,X1], dtype=tf.float32)

Y0, Y1 = output_seqs

#Let’s simplify this.
#The following code builds the same RNN again, but this time it takes a single input
#placeholder of shape [None, n_steps, n_inputs] where the first dimension is the
#mini-batch size. Then it extracts the list of input sequences for each time step. X_seqs
#is a Python list of n_steps tensors of shape [None, n_inputs], where once again the
#first dimension is the mini-batch size

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(
        basic_cell, X_seqs, dtype=tf.float32)
outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])

X_batch = np.array([
        # t = 0 t = 1
        [[0, 1, 2], [9, 8, 7]], # instance 0
        [[3, 4, 5], [0, 0, 0]], # instance 1
        [[6, 7, 8], [6, 5, 4]], # instance 2
        [[9, 0, 1], [3, 2, 1]], # instance 3
])
    
with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})
    
print()
