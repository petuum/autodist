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

    print('I am going to a scope.')
    with tf.Graph().as_default(), autodist.scope():
        x = tf.compat.v1.placeholder(shape=[NUM_EXAMPLES], dtype=tf.float32)

        W = tf.Variable(5.0, name='W')
        b = tf.Variable(0.0, name='b')

        def train_step(x):

            def y(x):
                return W * x + b

            def l(predicted_y, desired_y):
                return tf.reduce_mean(tf.square(predicted_y - desired_y))

            optimizer = tf.optimizers.SGD(0.01)
            # optimizer.iterations = get_or_create_global_step()

            with tf.GradientTape() as tape:
                loss = l(y(x), outputs)
                vs = [W, b]

                # gradients = tape.gradient(target=loss, sources=vs)
                gradients = tf.gradients(loss, vs)

                train_op = optimizer.apply_gradients(zip(gradients, vs))
                print(optimizer.iterations)
            return loss, train_op, optimizer.iterations, b

        symbol = train_step(x)
        sess = autodist.create_distributed_session()
        for epoch in range(EPOCHS):
            l, t, i, b = sess.run(fetches=symbol, feed_dict={x: inputs})
            print('node: {}, step: {}, loss: {}\nb:{}'.format(autodist._cluster.get_local_address(), i, l, b))

    print('I am out of scope')


