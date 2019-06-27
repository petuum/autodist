import sys
import os
import json

import numpy as np
from autodist.ps_strategy import PSStrategy
import tensorflow as tf

from tensorflow.python.training.training_util import get_or_create_global_step

file = os.path.join(os.path.dirname(__file__), 'cluster_spec.json')
cluster_spec = tf.train.ClusterSpec(json.load(open(file)))


def main(_):
    TRUE_W = 3.0
    TRUE_b = 2.0
    NUM_EXAMPLES = 1000
    EPOCHS = 10

    inputs = np.random.randn(NUM_EXAMPLES)
    noises = np.random.randn(NUM_EXAMPLES)
    outputs = inputs * TRUE_W + TRUE_b + noises

    d = PSStrategy(cluster_spec)

    class MyIterator:

        def initialize(self):
            return tf.zeros(1)

        def get_next(self):
            # a fake one
            return inputs

    inputs_iterator = MyIterator()

    with d.scope():
        # x = placeholder(shape=[NUM_EXAMPLES], dtype=tf.float32)

        def train_step(input):
            W = tf.Variable(5.0)
            b = tf.Variable(0.0)

            def y(x):
                return W * x + b

            def l(predicted_y, desired_y):
                return tf.reduce_mean(tf.square(predicted_y - desired_y))

            optimizer = tf.optimizers.SGD(0.01)
            optimizer.iterations = get_or_create_global_step()

            with tf.GradientTape() as tape:
                loss = l(y(input), outputs)
                vs = [W, b]

                gradients = tape.gradient(target=loss, sources=vs)
                train_op = optimizer.apply_gradients(zip(gradients, vs))
            return loss, train_op

        for epoch in range(EPOCHS):
            loss, _ = d.experimental_run(train_step, inputs_iterator)
            print(f"train_loss: {loss}")


main(sys.argv)
