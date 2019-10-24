import os
import numpy as np
import tensorflow as tf

from tensorflow.python.training.training_util import get_or_create_global_step
from autodist.const import Env


def main(autodist):

    TRUE_W = 3.0
    TRUE_b = 2.0
    NUM_EXAMPLES = 1000
    EPOCHS = 2

    # For Integration Value Test: (For more information, check the assertions below)
    seed = 456 if bool(os.environ.get(Env.AUTODIST_WORKER.name)) else 123
    np.random.seed(seed)

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
            optimizer.iterations = get_or_create_global_step()

            with tf.GradientTape() as tape:
                loss = l(y(input), outputs)
                vs = [W, b]

                # gradients = tape.gradient(target=loss, sources=vs)
                gradients = tf.gradients(loss, vs)

                train_op = optimizer.apply_gradients(zip(gradients, vs))
            return loss, train_op, b

        assert EPOCHS == 2
        for epoch in range(EPOCHS):
            l_val, _, b_val = train_step(input=inputs_iterator.get_next())
            print('loss:', l_val)

        # Integration Value Test:
        # It requires np.random.seed to be 123 on chief, 456 on worker.
        # Variable b's gradients[1] == -4.17503 on chief, == -4.05530 on worker
        num_workers = len(autodist._cluster.cluster_spec['worker'])
        # When SGD learning rate == 0.01 and b is initialied by zero,
        # the updated variable b value can be verified as below
        if num_workers == 1:
            assert np.allclose(b_val, 0.01 * 4.17503)
        elif num_workers == 2:
            # Between-graph dense conditional accumulator average verification
            assert np.allclose(b_val, 0.01 * (4.17503 + 4.05530) / 2)
            # TODO: between graph sparse verification
