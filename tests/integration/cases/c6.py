import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.util import nest


def main(autodist):
    tf.compat.v1.disable_control_flow_v2()

    def get_lstm(scope, units, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(
                num_units=units,
                use_peepholes=True,
                dtype=tf.float32,
                reuse=reuse,
                name=scope + '/LSTM')
            return lstm_cell

    def create_dynamic_lstm(batch_size, state_size, max_steps):
        cell = get_lstm('SL', state_size)
        y = tf.constant(np.random.rand(1, state_size), dtype=tf.float32)
        qq = tf.Variable(tf.zeros([5, 5], tf.float32), name="QQ")

        def dynamic_lstm_input_fn(batch_size, state_size, max_steps):
            # We make inputs and sequence_length constant so that multiple session.run
            # calls produce the same result.
            inputs = tf.constant(np.random.rand(batch_size, max_steps, state_size), dtype=dtypes.float32)
            sequence_length = np.random.randint(0, size=[batch_size], high=max_steps + 1)
            sequence_length = tf.constant(sequence_length, dtype=dtypes.int32)
            return inputs, sequence_length

        inputs, sequence_length = dynamic_lstm_input_fn(batch_size, state_size, max_steps)

        zeros = tf.zeros([state_size])

        def loop_fn(i):
            inputs_ta = tf.TensorArray(
                dtypes.float32, size=max_steps, element_shape=[batch_size, state_size])
            inputs_time_major = tf.transpose(inputs, [1, 0, 2])
            inputs_ta = inputs_ta.unstack(inputs_time_major)
            sequence_length_i = tf.gather(sequence_length, i)

            def body_fn(t, state, ta):
                inputs_t = tf.expand_dims(
                    tf.gather(inputs_ta.read(t), i), 0)
                output, new_state = cell(inputs_t, state)
                output = tf.reshape(output, [-1])
                done = t >= sequence_length_i
                output = tf.where(done, zeros, output)
                new_state = [tf.where(done, s, ns) for s, ns in
                             zip(nest.flatten(state), nest.flatten(new_state))]
                new_state = nest.pack_sequence_as(state, new_state)
                return t + 1, new_state, ta

            def condition_fn(t, _, unused):
                del unused
                return t < max_steps

            initial_state = cell.zero_state(1, dtypes.float32)

            _, state, ta = tf.while_loop(condition_fn, body_fn, [
                0, initial_state,
                tf.TensorArray(dtypes.float32, max_steps)
            ])

            new_state = [tf.reshape(x, [-1]) for x in nest.flatten(state)]
            new_state = nest.pack_sequence_as(initial_state, new_state)

            return ta.stack(), new_state

        major_version, _, _ = tf.version.VERSION.split('.')
        if major_version == '1':
            # self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        else:
            # self.optimizer = tf.optimizers.Adagrad(learning_rate=0.1, initial_accumulator_value=1.0)
            optimizer = tf.optimizers.Adam(learning_rate=0.1)

        outputs = []
        for t in range(3):
            _, o = loop_fn(t)
            outputs.append(o)
        output1 = tf.expand_dims(tf.reduce_mean(tf.concat(outputs, axis=0), axis=0), axis=0)
        output = tf.matmul(output1, qq)
        var_list = variables.trainable_variables() + ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
        loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
        grads = tf.gradients(loss, var_list)
        train_op = optimizer.apply_gradients(zip(grads, var_list))
        return train_op, output

    with tf.Graph().as_default(), autodist.scope():
        train_op, output = create_dynamic_lstm(6, 5, 2)
        sess = autodist.create_distributed_session()
        _, output_v = sess.run([train_op, output])
        print(output_v)
