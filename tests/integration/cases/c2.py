import time
import tensorflow as tf
import numpy as np


def main(autodist):
    vocab_size = 10000
    embedding_size = 16
    hidden_dim = 16
    max_steps = 60
    batch_size = 128
    log_frequency = 10

    class SimpleModel:
        def __init__(self):
            self.emb1 = tf.Variable(tf.random.uniform([vocab_size, embedding_size]),
                                   name='emb1',
                                   trainable=True,
                                   dtype=tf.float32)
            self.emb2 = tf.Variable(tf.random.uniform([vocab_size, embedding_size]),
                                   name='emb2',
                                   trainable=True,
                                   dtype=tf.float32)
            self.w1 = tf.Variable(tf.random.uniform([embedding_size, hidden_dim]),
                                  name='w1',
                                  trainable=True,
                                  dtype=tf.float32)
            self.b1 = tf.Variable(tf.zeros([hidden_dim]),
                                  name='b1',
                                  trainable=True,
                                  dtype=tf.float32)
            self.w2 = tf.Variable(tf.random.uniform([hidden_dim, 1]),
                                  name='w2',
                                  trainable=True,
                                  dtype=tf.float32)
            self.b2 = tf.Variable(tf.zeros([1]),
                                  name='b2',
                                  trainable=True,
                                  dtype=tf.float32)
            major_version, _, _ = tf.version.VERSION.split('.')
            if major_version == '1':
                # self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.2)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=0.2)
            else:
                # self.optimizer = tf.optimizers.Adagrad(learning_rate=0.2, initial_accumulator_value=1.0)
                self.optimizer = tf.optimizers.Adam(learning_rate=0.2)

        def forward(self, x, y):
            # embedding layer
            x, z = tf.nn.embedding_lookup(self.emb1, x), tf.nn.embedding_lookup(self.emb2, x)
            # Conditional (so we don't partition self.emb2)
            z = tf.cond(tf.reduce_sum(z) > 0, lambda: tf.identity(z), lambda: tf.identity(z))
            # Combine
            x = .5 * (x + z)
            # global average pool
            x = tf.math.reduce_mean(x, axis=1)
            # dense
            x = tf.linalg.matmul(x, self.w1) + self.b1
            x = tf.nn.relu(x)
            logits = tf.linalg.matmul(x, self.w2) + self.b2  # logits
            logits = tf.squeeze(logits)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(loss)
            return loss

        @autodist.function
        def train_fn(self, xy):
            x, y = xy
            # with tf.GradientTape() as tape:
            loss = self.forward(x, y)
            trainables = [self.emb1, self.emb2, self.w1, self.b1, self.w2, self.b2]
            gradients = tf.gradients(loss, trainables)
            # gradients = tape.gradient(loss, trainables)
            # strategy requires users to provide the train_op handle
            train_op = self.optimizer.apply_gradients(zip(gradients, trainables))
            return loss, train_op, self.emb1, self.emb2

    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                               value=0,
                                                               padding='post',
                                                               maxlen=256)
    test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                              value=0,
                                                              padding='post',
                                                              maxlen=256)
    train_labels = train_labels.astype(np.float32)
    with tf.Graph().as_default(), autodist.scope():  # AutoDist code
        my_iterator = tf.compat.v1.data.Dataset.from_tensor_slices((train_data, train_labels)) \
            .shuffle(25000).batch(batch_size).repeat(1).make_one_shot_iterator().get_next()
        # The fix https://github.com/tensorflow/tensorflow/pull/34295 is only included after TensorFlow > 2.1
        # e.g in TensorFlow 2.2.0rc1 or after
        # Before that, we never touch the end of iterator in this test case with limited steps.
        # my_iterator = MyIterator().get_next()
        model = SimpleModel()
        prev_time = time.time()
        for local_step in range(max_steps):
            # fetch train_op and loss
            loss, _, _, _ = model.train_fn(my_iterator)
            # loss, _ = autodist.run(model.train_fn, my_iterator)
            if local_step % log_frequency == 0:
                cur_time = time.time()
                elapsed_time = cur_time - prev_time
                num_sentences = batch_size * log_frequency
                wps = float(num_sentences) / elapsed_time
                print("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f" % (
                    local_step, cur_time - prev_time, wps, loss))
                prev_time = cur_time
