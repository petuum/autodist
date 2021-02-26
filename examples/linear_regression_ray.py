import sys
import os
import time

import numpy as np
import tensorflow as tf

from autodist import AutoDist
from autodist.strategy import PS, PSLoadBalancing, PartitionedPS, AllReduce, Parallax

import ray
from autodist.ray import TFTrainer, TFRunner

ray.init(address='auto')

EPOCHS = 10

def data_creator():
    TRUE_W = 3.0
    TRUE_b = 2.0
    NUM_EXAMPLES = 1000

    inputs = np.random.randn(NUM_EXAMPLES)
    noises = np.random.randn(NUM_EXAMPLES)
    outputs = inputs * TRUE_W + TRUE_b + noises

    class MyIterator:
        def initialize(self):
            return tf.zeros(1)
        def get_next(self):
            # a fake one
            return inputs
    return MyIterator().get_next(), outputs


class Model:
    def __init__(self):
        self.W = tf.Variable(5.0, name='W', dtype=tf.float64)
        self.b = tf.Variable(0.0, name='b', dtype=tf.float64)

    def __call__(self, x):
        return self.W * x + self.b


def train_step(model, inputs, outputs):
    def l(predicted_y, desired_y):
        return tf.reduce_mean(tf.square(predicted_y - desired_y))

    major_version, _, _ = tf.version.VERSION.split('.')
    if major_version == '1':
        optimizer = tf.train.GradientDescentOptimizer(0.01)
    else:
        optimizer = tf.optimizers.SGD(0.01)

    loss = l(model(inputs), outputs)
    vs = [model.W, model.b]

    gradients = tf.gradients(loss, vs)

    train_op = optimizer.apply_gradients(zip(gradients, vs))
    return loss, train_op, model.b


def main(_):
    trainer = TFTrainer(Model, data_creator, train_step, PS())
    for epoch in range(EPOCHS):
        trainer.train()

    trainer.shutdown()

main(sys.argv)

