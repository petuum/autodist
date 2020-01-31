import numpy as np
import tensorflow as tf


def main(autodist):

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

    with tf.Graph().as_default() as g, autodist.scope():
        W = tf.Variable(5.0, name='W', dtype=tf.float64)
        b = tf.Variable(0.0, name='b', dtype=tf.float64)

        inputs_iterator = MyIterator()

        def l(predicted_y, desired_y):
            return tf.reduce_mean(tf.square(predicted_y - desired_y))

        def f(input):
            def body(step, state):
                new_state = tf.nn.sigmoid(W * state + b)
                return step + 1, new_state

            def condition(step, _):
                return step < 3

            final_step, final_x = tf.while_loop(condition, body, [0, input])
            return final_x

        @autodist.function
        def train_step(input):
            major_version, _, _ = tf.version.VERSION.split('.')
            if major_version == '1':
                optimizer = tf.train.GradientDescentOptimizer(0.01)
            else:
                optimizer = tf.optimizers.SGD(0.01)
            loss = l(f(input), outputs)
            vs = [W, b]
            # gradients = tape.gradient(target=loss, sources=vs)
            gradients = tf.gradients(loss, vs)
            train_op = optimizer.apply_gradients(zip(gradients, vs))
            return loss, train_op, b

        for epoch in range(EPOCHS):
            l, t, b = train_step(input=inputs_iterator.get_next())
            print('node: {}, loss: {}\nb:{}'.format(autodist._cluster.get_local_address(), l, b))
