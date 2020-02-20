import os
import numpy as np
import tensorflow as tf

from autodist.const import ENV


def main(autodist):

    TRUE_W = 3.0
    TRUE_b = 2.0
    NUM_EXAMPLES = 1000
    EPOCHS = 1

    # For Integration Value Test: (For more information, check the assertions below)
    seed = 456 if bool(ENV.AUTODIST_WORKER.val) else 123
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
    with tf.Graph().as_default(), autodist.scope():
        x = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
        y = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)

        W = tf.Variable(5.0, name='W')
        b = tf.Variable(0.0, name='b')

        def train_step(x):

            def f(x):
                return W * x + b

            def l(predicted_y, desired_y):
                return tf.reduce_mean(tf.square(predicted_y - desired_y))

            major_version, _, _ = tf.version.VERSION.split('.')
            if major_version == '1':
                optimizer = tf.train.GradientDescentOptimizer(0.01)
            else:
                optimizer = tf.optimizers.SGD(0.01)

            with tf.GradientTape() as tape:
                loss = l(f(x), y)
                vs = [W, b]

                # gradients = tape.gradient(target=loss, sources=vs)
                gradients = tf.gradients(loss, vs)

                train_op = optimizer.apply_gradients(zip(gradients, vs))
            return loss, train_op, b

        assert EPOCHS == 1
        fetches = train_step(x)
        session = autodist.create_distributed_session()
        for epoch in range(EPOCHS):
            l_val, _, _ = session.run(fetches=fetches, feed_dict={x: inputs_iterator.get_next(), y: outputs})
            print('loss:', l_val)
            # Seperate the fetches of var to guarantee the state
            b_val = session.run(b)

        if getattr(autodist._strategy_builder, '_sync', True):
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
