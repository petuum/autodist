# Copyright 2021 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import os
import time
import ray
import numpy as np
import tensorflow as tf

from autodist.strategy import PS, PSLoadBalancing, PartitionedPS, AllReduce, Parallax
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
    trainer = TFTrainer(PS(), Model, data_creator, train_step)
    for epoch in range(EPOCHS):
        trainer.train()

    trainer.shutdown()

main(sys.argv)

