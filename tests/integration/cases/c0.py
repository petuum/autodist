import os
import numpy as np
import tensorflow as tf

from autodist.const import ENV
from autodist.checkpoint.saver import Saver
from autodist.strategy import AllReduce, Parallax, PartitionedAR


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
        saver = Saver([W, b])
        session = autodist.create_distributed_session()
        for epoch in range(EPOCHS):
            l_val, _, _ = session.run(fetches=fetches, feed_dict={x: inputs_iterator.get_next(), y: outputs})
            print('loss:', l_val)
            # Seperate the fetches of var to guarantee the state
            W_val, b_val = session.run([W, b])

        # Try to save the two variables
        checkpoint_dir = '/tmp/ckpt/'
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_suffix = 'c0'
        checkpoint_name = checkpoint_dir + checkpoint_suffix
        saver.save(session, checkpoint_name, global_step=epoch)
        print('Checkpoint saved at {%s}' % checkpoint_name)

        num_workers = len(autodist._cluster.cluster_spec['worker'])
        if getattr(autodist._strategy_builder, '_sync', True):
            # Integration Value Test:
            # It requires np.random.seed to be 123 on chief, 456 on worker.
            # Variable b's gradients[1] == -4.17503 on chief, == -4.05530 on worker
            # When SGD learning rate == 0.01 and b is initialied by zero,
            # the updated variable b value can be verified as below
            if num_workers == 1:
                assert np.allclose(b_val, 0.01 * 4.17503)
            elif num_workers == 2:
                # Between-graph dense conditional accumulator average verification

                def get_gpu_distribution(resource_spec):
                    """A list contains the distribution of devices."""
                    gpu_dist = dict()
                    for gpu_str, _ in resource_spec.gpu_devices:
                        ip = gpu_str.split(':')[0]
                        if ip in gpu_dist:
                            gpu_dist[ip] += 1
                        else:
                            gpu_dist[ip] = 1
                    return list(gpu_dist.values())

                dist = get_gpu_distribution(autodist._resource_spec)
                # if uneven gpu distribution, allreduce grad is a weighted sum.
                if dist[0] != dist[1] and \
                        (isinstance(autodist._strategy_builder, AllReduce) or
                         isinstance(autodist._strategy_builder, Parallax) or
                         isinstance(autodist._strategy_builder, PartitionedAR)):
                    assert np.allclose(b_val, 0.01 * (4.17503 * dist[0] + 4.05530 * dist[1]) / (dist[0] + dist[1]))
                else:
                    assert np.allclose(b_val, 0.01 * (4.17503 + 4.05530) / 2)
                # TODO: between graph sparse verification

        # check the checkpoint existence
        checkpoint = checkpoint_name + '-' + str(epoch)
        assert(os.path.exists(checkpoint + '.meta')) # meta file
        assert(os.path.exists(checkpoint + '.index'))  # meta file
        assert(os.path.exists(checkpoint + '.data-00000-of-00001'))  # meta file
        print('Checkpoint {} exists.'.format(checkpoint))

        # Now check restoration and the correctness of the checkpoint
        tf_saver = tf.compat.v1.train.Saver([W, b])
        tf_sess = tf.compat.v1.Session()
        tf_saver.restore(tf_sess, tf.train.latest_checkpoint(checkpoint_dir))
        restore_W_val, restore_b_val = tf_sess.run([W, b])
        print('original value {} vs. restored value {}'.format(W_val, restore_W_val))
        print('original value {} vs. restored value {}'.format(b_val, restore_b_val))
        assert(np.allclose(b_val, restore_b_val))
        assert(np.allclose(W_val, restore_W_val))
