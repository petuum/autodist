import sys
import os

import numpy as np
import tensorflow as tf

from tensorflow.python.training.training_util import get_or_create_global_step

from autodist import AutoDist

resource_spec_file = os.path.join(os.path.dirname(__file__), 'resource_spec.yml')
config_file = os.path.join(os.path.dirname(__file__), 'runner_config.yml')


def main(_):
    autodist = AutoDist(resource_spec_file, 'PS', runner_config_file=config_file)

    TRUE_W = 3.0
    TRUE_b = 2.0
    NUM_EXAMPLES = 1000
    EPOCHS = 10

    inputs = np.random.randn(NUM_EXAMPLES)
    noises = np.random.randn(NUM_EXAMPLES)
    outputs = inputs * TRUE_W + TRUE_b + noises

    class MyIterator:

        def initialize(self):
            return tf.zeros(1)

        def get_next(self):
            # a fake one
            return inputs

    inputs_iterator = MyIterator()
    print('I am going to a scope.')
    with autodist.scope():
        # x = placeholder(shape=[NUM_EXAMPLES], dtype=tf.float32)

        W = tf.Variable(5.0, name='W', dtype=tf.float64)
        b = tf.Variable(0.0, name='b', dtype=tf.float64)

        @autodist.function
        def train_step(input):

            def y(x):
                return W * x + b

            def l(predicted_y, desired_y):
                return tf.reduce_mean(tf.square(predicted_y - desired_y))

            optimizer = tf.optimizers.SGD(0.01)
            # optimizer.iterations = get_or_create_global_step()

            with tf.GradientTape() as tape:
                loss = l(y(input), outputs)
                vs = [W, b]

                # gradients = tape.gradient(target=loss, sources=vs)
                gradients = tf.gradients(loss, vs)

                train_op = optimizer.apply_gradients(zip(gradients, vs))
                print(optimizer.iterations)
            return loss, train_op, optimizer.iterations, b

        for epoch in range(EPOCHS):
            l, t, i, b = train_step(input=inputs_iterator.get_next())
            print('node: {}, step: {}, loss: {}\nb:{}'.format(autodist._cluster.get_local_address(), i, l, b))

    print('I am out of scope')


main(sys.argv)

