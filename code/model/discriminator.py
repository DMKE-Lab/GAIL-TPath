# -*- coding: utf-8 -*-
from __future__ import division
import collections
import numpy as np
import sys
import tensorflow as tf
import time
import math
from BFS.BFS import BFS
from BFS.KB import KB
from itertools import count
from networks import discriminator_nn
LAMBDA = 5

class Discriminator(object):
    def __init__(self, batch_size, embedding_dim, learning_rate=0.001):
        self.initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('discriminator') as scope:
            self.task = tf.placeholder(tf.string)
            self.real_inputs = tf.placeholder(tf.float32, [batch_size, 1, embedding_dim], name='real_inputs')
            self.fake_inputs = tf.placeholder(tf.float32, [batch_size, 1, embedding_dim], name='fake_inputs')
            disc_real = discriminator_nn(self.real_inputs, embedding_dim, self.task, self.initializer)
            scope.reuse_variables()
            disc_fake = discriminator_nn(self.fake_inputs, embedding_dim, self.task, self.initializer)
            original_disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
            alpha = tf.random_uniform(shape=[batch_size, 1, 1], minval=0., maxval=1.)
            differences = self.fake_inputs - self.real_inputs
            interpolate = self.real_inputs + (alpha * differences)
            grad = \
            tf.gradients(discriminator_nn(interpolate, embedding_dim, self.task, self.initializer)[0], [interpolate])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), reduction_indices=[1, 2]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            self.disc_cost = (original_disc_cost + LAMBDA * gradient_penalty)
            self.gen_reward = tf.reduce_mean(disc_fake)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.disc_cost)

    def predict(self, real, fake, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run([self.disc_cost, self.gen_reward], {self.task: 'test', self.real_inputs: real, self.fake_inputs: fake})

    def update(self, real, fake, sess=None):
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.train_op, self.disc_cost],{self.task: 'train', self.real_inputs: real, self.fake_inputs: fake})
        return loss
