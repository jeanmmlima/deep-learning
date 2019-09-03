# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:40:53 2019

@author: Jean Mário
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial
import sys
import os

#data
mnist = input_data.read_data_sets("/tmp/data/")

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_outputs = 10

learning_rate = 0.01
l2_reg = 0.0005

activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
y = tf.placeholder(tf.int32, shape=[None])

weights1_init = initializer([n_inputs,n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])
weights3_init = initializer([n_hidden2, n_outputs])

#weights
weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")

#bieases
biases1 = tf.Variable(tf.zeros(n_hidden1), name= "biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name= "biases2")
biases3 = tf.Variable(tf.zeros(n_outputs), name= "biases3")

#layers
hidden1 = activation(tf.matmul(X,weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1,weights2) + biases2)
logits = tf.matmul(hidden2, weights3) + biases3

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
reg_loss = regularizer(weights1) + regularizer(weights2) + regularizer(weights3)
loss = cross_entropy + reg_loss
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
pretrain_saver = tf.train.Saver([weights1, weights2, biases1, biases2])
saver = tf.train.Saver() # not shown in the book


