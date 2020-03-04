import os
import numpy as np
import time
import tensorflow as tf


def main(autodist):

    TRUE_W = 3.0
    TRUE_b = 2.0
    NUM_EXAMPLES = 1000
    TATOL_STEPS = 10

    # Set non-chief node as the slow node while other nodes will need to wait for the slow node.
    is_slow_node = not autodist._cluster.is_chief()
    # It looks like git repo only prints messages on the chief node, and only 2-machine scenario is tested.
    # We need to print messages from fast nodes. So set the chief node as the fast node.
    sleep_every_steps = 3
    sleep_time = 5
    if len(autodist._cluster._task_to_address) <= 1:
        print("[Warning] Testing AutoDist's staleness requires more than 1 machine. "
              "Reduce to single machine strategy.")

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

        step2time = {}
        local_address = autodist._cluster.get_local_address()

        for step in range(TATOL_STEPS):
            l, t, b = session.run(fetches)
            step2time[step] = time.ctime().split()[-2]
            print("time={}, node={}, step={}, loss={}, b={}".format(
                step2time[step], local_address, step, l, b))

            if is_slow_node and (step+1) % sleep_every_steps == 0:
                print("node {} sleeps for {} seconds at step {}.".format(local_address, sleep_time, step))
                time.sleep(sleep_time)

    print('I am out of scope')

    if not is_slow_node:
        verify_staleness_on_fast_node(step2time, local_address, autodist._strategy_builder._staleness,
                                      sleep_every_steps, sleep_time)


def verify_staleness_on_fast_node(step2time, local_address, staleness, sleep_every_steps, sleep_time):
    """Verify stalenss on fast node."""
    valid = True
    print("Verifying staleness for local_address={}, staleness={}, sleep_every_steps={}, "
          "sleep_time={}".format(local_address, staleness, sleep_every_steps, sleep_time))

    def _str2sec(t):
        # convert string of time to total seconds.
        t = [int(i) for i in t.split(':')]
        assert len(t) == 3
        return t[0] * 3600 + t[1] * 60 + t[2]

    for step, time in step2time.items():
        print('step={}, time={}'.format(step, time))

    for i in range(staleness+sleep_every_steps-1):
        if not (_str2sec(step2time[i+1]) - _str2sec(step2time[i]) <= 1):  # at roughly the same time
            print("[STALENESS ERROR]: step {} and step {} have time gap.\n".format(i+1, i))
            valid = False

    for i in range(staleness+sleep_every_steps-1, len(step2time)-1):
        if (i-staleness-sleep_every_steps+1) % sleep_every_steps == 0:
            if not (_str2sec(step2time[i+1]) - _str2sec(step2time[i]) >= sleep_time - 1):
                print("[STALENESS ERROR]: step {} and step {} have no time gap.\n".format(i+1, i))
                valid = False
            # account for rum time for steps within staleness time.
        else:
            if not (_str2sec(step2time[i+1]) - _str2sec(step2time[i]) <= 1): # at roughly the same time
                print("[STALENESS ERROR]: step {} and step {} have time gap.\n".format(i+1, i))
                valid = False

    if valid:
        print("Staleness has been successfully verified.\n")
    else:
        print("Staleness verification has failed.\n")
