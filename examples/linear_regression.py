import sys
import os

import numpy as np
import tensorflow as tf


from autodist import AutoDist
from autodist.strategy import PS, PSLoadBalancing, PartitionedPS, AllReduce, Parallax
from autodist.strategy.auto_strategy import AutoStrategy

resource_spec_file = os.path.join(os.path.dirname(__file__), 'resource_spec.yml')


def main(_):
    autodist = AutoDist(resource_spec_file, AllReduce(128))
    # autodist = AutoDist(resource_spec_file, AutoStrategy())

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
    with tf.Graph().as_default() as g, autodist.scope():
        # x = placeholder(shape=[NUM_EXAMPLES], dtype=tf.float32)

        W = tf.Variable(5.0, name='W', dtype=tf.float64)
        b = tf.Variable(0.0, name='b', dtype=tf.float64)

        def train_step(input):

            def y(x):
                return W * x + b

            def l(predicted_y, desired_y):
                return tf.reduce_mean(tf.square(predicted_y - desired_y))

            major_version, _, _ = tf.version.VERSION.split('.')
            if major_version == '1':
                optimizer = tf.train.GradientDescentOptimizer(0.01)
            else:
                optimizer = tf.optimizers.SGD(0.01)

            with tf.GradientTape() as tape:
                loss = l(y(input), outputs)
                vs = [W, b]

                # gradients = tape.gradient(target=loss, sources=vs)
                gradients = tf.gradients(loss, vs)

                train_op = optimizer.apply_gradients(zip(gradients, vs))
            return loss, train_op, b

        fetches = train_step(inputs_iterator.get_next())
        session = autodist.create_distributed_session()
        for epoch in range(EPOCHS):
            l, t, b = session.run(fetches)
            print('node: {}, loss: {}\nb:{}'.format(autodist._cluster.get_local_address(), l, b))

    print('I am out of scope')


main(sys.argv)
