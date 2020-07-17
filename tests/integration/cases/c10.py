import os
import numpy as np
import tensorflow as tf

from autodist.const import ENV, ONLY_MASTER_SAVE
from autodist.checkpoint.saver import Saver
from autodist.strategy import AllReduce, Parallax, PartitionedAR, RandomAxisPartitionAR


def main(autodist):
    # Test saver on NFS system

    TRUE_W = 3.0
    TRUE_b = 2.0
    NUM_EXAMPLES = 1000
    EPOCHS = 1

    seed = 456 if bool(ENV.AUTODIST_WORKER.val) else 123
    np.random.seed(seed)

    inputs = np.random.randn(NUM_EXAMPLES)
    noises = np.random.randn(NUM_EXAMPLES)
    outputs = inputs * TRUE_W + TRUE_b + noises

    class MyIterator:

        def initialize(self):
            return tf.zeros(1)

        def get_next(self):
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
                gradients = tf.gradients(loss, vs)
                train_op = optimizer.apply_gradients(zip(gradients, vs))
            return loss, train_op, b

        assert EPOCHS == 1
        fetches = train_step(x)
        saver = Saver([W, b])
        session = autodist.create_distributed_session()
        for epoch in range(EPOCHS):
            l_val, _, _ = session.run(fetches=fetches, feed_dict={x: inputs_iterator.get_next(), y: outputs})
            print('loss:', l_val)
            # Seperate the fetches of var to guarantee the state
            W_val, b_val = session.run([W, b])

        # Try to save the two variables
        checkpoint_dir = '/tmp/ckpt_c10/'
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        # Only save the model on master node if autodist is used with NFS.
        checkpoint_suffix = 'c10'
        checkpoint_name = checkpoint_dir + checkpoint_suffix
        if ONLY_MASTER_SAVE:
            saver.save(session, checkpoint_name, global_step=epoch)
            print('Checkpoint saved at {%s}' % checkpoint_name)
        else:
            print("Skip saving on worker nodes.")

        # check the checkpoint existence only on master node
        checkpoint = checkpoint_name + '-' + str(epoch)
        if autodist.IS_AUTODIST_CHIEF:
            assert(os.path.exists(checkpoint + '.meta')) # meta file
            assert(os.path.exists(checkpoint + '.index'))  # meta file
            assert(os.path.exists(checkpoint + '.data-00000-of-00001'))  # meta file
            print('Checkpoint {} exists which saved by master.'.format(checkpoint))
        else:
            assert(not os.path.exists(checkpoint + '.meta')) # meta file
            assert(not os.path.exists(checkpoint + '.index'))  # meta file
            assert(not os.path.exists(checkpoint + '.data-00000-of-00001'))  # meta file
            print("Checkpoint saving skipped on worker nodes confirmed.")
